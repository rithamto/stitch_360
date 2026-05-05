import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:exif/exif.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:path/path.dart' as path;
import 'package:room_scan/features/scan_raw/domain/entities/entities.dart';
import 'package:room_scan/features/scan_raw/domain/repositories/repositories.dart';
import 'package:room_scan/routes/app_pages.dart';

/// State machine for the 360° scanning process.
enum ScanState { initializing, ready, scanning, completed, error }

/// Controller managing the full 360° panorama capture flow.
///
/// Supports two grid densities:
///   - **Main camera** (1x, HFOV ≈ 65°): 52-point grid across 7 rows
///   - **Wide-angle camera** (0.5x, HFOV ≈ 120°): 22-point grid across 5 rows
///
/// Auto-captures when the device is held stable and aligned with a target cell.
class ScanRawController extends GetxController {
  ScanRawController({required IPanoramaRepository repository})
    : _repository = repository;

  final IPanoramaRepository _repository;
  CameraController? cameraController;

  // ── Grid configurations ────────────────────────────────────────────

  /// Main camera grid (1x): 40 points across 5 rows.
  /// Designed for HFOV ≈ 65° with ≥30% overlap between adjacent frames.
  ///
  ///   Row 0  Zenith (+90°) :  1 point   (single cap)
  ///   Row 1  Top    (+35°) : 12 points  (staggered vs horizon)
  ///   Row 2  Horizon ( 0°) : 14 points  (densest, reference)
  ///   Row 3  Bottom (-35°) : 12 points  (staggered vs horizon)
  ///   Row 4  Nadir  (-90°) :  1 point   (single cap)
  static const List<Map<String, dynamic>> _mainCameraGrid = [
    {'pitch': 90.0,  'count': 1,  'yawOffset': 0.0},
    {'pitch': 35.0,  'count': 12, 'yawOffset': 15.0},
    {'pitch': 0.0,   'count': 14, 'yawOffset': 0.0},
    {'pitch': -35.0, 'count': 12, 'yawOffset': 15.0},
    {'pitch': -90.0, 'count': 1,  'yawOffset': 0.0},
  ];

  /// Wide-angle camera grid (0.5x): 30 points across 5 rows.
  /// Matches the sample app layout for HFOV ≈ 120°.
  ///
  ///   Row 0  Zenith (+90°) :  1 point
  ///   Row 1  Top    (+45°) :  8 points  (staggered vs row 2)
  ///   Row 2  Horizon ( 0°) : 12 points  (reference)
  ///   Row 3  Bottom (-45°) :  8 points  (staggered vs row 2)
  ///   Row 4  Nadir  (-90°) :  1 point
  static const List<Map<String, dynamic>> _wideCameraGrid = [
    {'pitch': 90.0,  'count': 1,  'yawOffset': 0.0},
    {'pitch': 45.0,  'count': 8,  'yawOffset': 22.5},
    {'pitch': 0.0,   'count': 12, 'yawOffset': 0.0},
    {'pitch': -45.0, 'count': 8,  'yawOffset': 22.5},
    {'pitch': -90.0, 'count': 1,  'yawOffset': 0.0},
  ];

  /// Whether the wide-angle camera is currently active.
  final isWideAngle = false.obs;

  /// The currently active grid layout (reactive).
  List<Map<String, dynamic>> get activeGridLayout =>
      isWideAngle.value ? _wideCameraGrid : _mainCameraGrid;

  static const double _minCoverage = 0.95;

  // ── Observables ────────────────────────────────────────────────────
  final scanState = ScanState.initializing.obs;

  final yaw = 0.0.obs;
  final pitch = 0.0.obs;
  final roll = 0.0.obs;
  final isStable = false.obs;
  final rotationSpeed = 0.0.obs;

  final coveragePercent = 0.0.obs;
  final capturedFrames = <RawFrame>[].obs;
  final errorMessage = ''.obs;

  // Guided target
  final targetRow = 0.obs;
  final targetCol = 0.obs;

  /// Sphere grid: true = captured.
  /// Each inner list length matches activeGridLayout[row]['count'].
  late final RxList<RxList<bool>> sphereGrid;

  // ── Internal ───────────────────────────────────────────────────────
  StreamSubscription<DeviceOrientationData>? _orientationSub;
  Timer? _stableTimer;
  bool _isCapturing = false;
  DateTime? _stableStartTime;
  double _lastYaw = 0.0;
  double _lastPitch = 0.0;

  late Directory sessionDir;

  /// Index of the last row in the active grid.
  int get _lastRowIndex => activeGridLayout.length - 1;

  // ── Lifecycle ──────────────────────────────────────────────────────

  @override
  void onInit() {
    super.onInit();
    _buildSphereGrid();
    _updateNextTarget();
    _initializeSession();
  }

  @override
  void onClose() {
    _orientationSub?.cancel();
    _stableTimer?.cancel();
    cameraController?.dispose();
    super.onClose();
  }

  // ── Grid management ────────────────────────────────────────────────

  /// (Re)builds the sphere grid to match the active layout.
  void _buildSphereGrid() {
    final layout = activeGridLayout;
    sphereGrid = RxList.generate(
      layout.length,
      (r) => RxList.generate(layout[r]['count'] as int, (_) => false),
    );
  }

  /// Switches between main camera and wide-angle camera grids.
  /// Resets the grid so the user starts a fresh capture with the new density.
  void switchCameraGrid({required bool wideAngle}) {
    if (isWideAngle.value == wideAngle) return;
    isWideAngle.value = wideAngle;
    _buildSphereGrid();
    capturedFrames.clear();
    coveragePercent.value = 0.0;
    _updateNextTarget();
  }

  // ── Initialize Camera + IMU ────────────────────────────────────────

  Future<void> _initializeSession() async {
    try {
      scanState.value = ScanState.initializing;

      // Initialize Camera
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw Exception('No camera found');
      final backCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      cameraController = CameraController(
        backCamera,
        ResolutionPreset.max,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await cameraController!.initialize();

      // Initialize IMU
      await _repository.initialize();

      // Prepare storage
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final tempDir = Directory.systemTemp;
      sessionDir = Directory('${tempDir.path}/panorama_$timestamp');
      await sessionDir.create();

      // Listen to orientation stream
      _orientationSub = _repository.orientationStream.listen(
        _onOrientationUpdate,
        onError: (e) {
          errorMessage.value = 'IMU error: $e';
        },
      );

      scanState.value = ScanState.scanning;
    } catch (e) {
      errorMessage.value = e.toString();
      scanState.value = ScanState.error;
    }
  }

  // ── Orientation updates ────────────────────────────────────────────

  void _onOrientationUpdate(DeviceOrientationData data) {
    yaw.value = data.yaw;
    pitch.value = data.pitch;
    roll.value = data.roll;
    isStable.value = data.isStable;

    // Calculate rotation speed for "Slow Down" warning
    final double dy = (data.yaw - _lastYaw).abs();
    final double dp = (data.pitch - _lastPitch).abs();
    rotationSpeed.value = dy > 180 ? (360 - dy) : dy + dp;

    _lastYaw = data.yaw;
    _lastPitch = data.pitch;

    if (scanState.value != ScanState.scanning) return;

    // Determine current grid cell
    final row = _pitchToRow(data.pitch);
    final col = _yawToCol(data.yaw, row);

    // CRITICAL: If current target is already captured, stop to prevent duplicates
    if (sphereGrid[targetRow.value][targetCol.value]) {
      _stableStartTime = null;
      return;
    }

    // Auto-capture criteria:
    // 1. Must be the target cell
    // 2. Must be stable
    // 3. For Zenith/Nadir: zone-based snap
    final isPolarTarget =
        (targetRow.value == 0 || targetRow.value == _lastRowIndex);
    bool isAligned = false;

    if (isPolarTarget) {
      if (targetRow.value == 0 && data.pitch > 75) isAligned = true;
      if (targetRow.value == _lastRowIndex && data.pitch < -75) {
        isAligned = true;
      }
    } else {
      isAligned = (row == targetRow.value && col == targetCol.value);
    }

    if (isAligned && data.isStable && !_isCapturing) {
      _stableStartTime ??= DateTime.now();
      final elapsed =
          DateTime.now().difference(_stableStartTime!).inMilliseconds;

      final requiredElapsed = isPolarTarget ? 200 : 600;
      if (elapsed >= requiredElapsed) {
        _captureCurrentFrame(
          data.yaw,
          data.pitch,
          targetRow.value,
          targetCol.value,
        );
      }
    } else if (!data.isStable || !isAligned) {
      _stableStartTime = null;
    }
  }

  // ── Capture logic ─────────────────────────────────────────────────

  Future<void> _captureCurrentFrame(
    double captureYaw,
    double capturePitch,
    int row,
    int col,
  ) async {
    if (_isCapturing || cameraController == null) return;
    _isCapturing = true;

    try {
      // 1. Capture Picture
      final xFile = await cameraController!.takePicture();
      final File imgFile = File(xFile.path);
      final Uint8List bytes = await imgFile.readAsBytes();

      // 2. Read EXIF Data for BE Stitching
      final Map<String, IfdTag> exifData = await readExifFromBytes(bytes);
      double iso = 0.0;
      double exposure = 0.0;
      double aperture = 0.0;

      try {
        if (exifData.containsKey('EXIF ISOSpeedRatings')) {
          iso = double.tryParse(
                exifData['EXIF ISOSpeedRatings']!.toString(),
              ) ??
              0.0;
        }
        if (exifData.containsKey('EXIF ExposureTime')) {
          final String expStr = exifData['EXIF ExposureTime']!.toString();
          if (expStr.contains('/')) {
            final parts = expStr.split('/');
            exposure = double.parse(parts[0]) / double.parse(parts[1]);
          } else {
            exposure = double.tryParse(expStr) ?? 0.0;
          }
        }
        if (exifData.containsKey('EXIF FNumber')) {
          final String apStr = exifData['EXIF FNumber']!.toString();
          if (apStr.contains('/')) {
            final parts = apStr.split('/');
            aperture = double.parse(parts[0]) / double.parse(parts[1]);
          } else {
            aperture = double.tryParse(apStr) ?? 0.0;
          }
        }
      } catch (e) {
        print("EXIF extraction error: $e");
      }

      // 3. Intrinsic Matrix Estimation
      final double hFov = isWideAngle.value ? 120.0 : 65.0;

      final ui.Codec codec = await ui.instantiateImageCodec(bytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      final double width = frameInfo.image.width.toDouble();
      final double height = frameInfo.image.height.toDouble();

      final double fx = width / (2 * tan((hFov / 2) * pi / 180.0));
      final double fy = fx;
      final double cx = width / 2.0;
      final double cy = height / 2.0;

      final int timestamp = DateTime.now().millisecondsSinceEpoch;

      final String destPath =
          '${sessionDir.path}/frame_${capturedFrames.length}'
          '_y${captureYaw.toStringAsFixed(1)}'
          '_p${capturePitch.toStringAsFixed(1)}'
          '_r${roll.value.toStringAsFixed(1)}'
          '_f${fx.toStringAsFixed(1)}'
          '_t$timestamp.jpg';

      await imgFile.copy(destPath);
      await imgFile.delete();

      final frame = RawFrame(
        index: capturedFrames.length,
        filePath: destPath,
        yaw: captureYaw,
        pitch: capturePitch,
        roll: roll.value,
        focalLength: fx,
        capturedAt: DateTime.now(),
        iso: iso,
        exposureTime: exposure,
        aperture: aperture,
        fx: fx,
        fy: fy,
        cx: cx,
        cy: cy,
      );

      capturedFrames.add(frame);

      // Lock focus after first frame to maintain sharp depth consistency,
      // but keep exposure automatic for OpenCV compensation.
      if (capturedFrames.length == 1) {
        try {
          await cameraController!.setFocusMode(FocusMode.auto);
          await cameraController!.setExposureMode(ExposureMode.auto);
        } catch (e) {
          print("Failed to lock focus: $e");
        }
      }

      sphereGrid[row][col] = true;
      sphereGrid.refresh();
      _updateCoverage();
      _updateNextTarget();

      HapticFeedback.mediumImpact();
    } catch (e) {
      print("Capture failed: $e");
    } finally {
      _isCapturing = false;
      _stableStartTime = null;
    }
  }

  /// Manual capture trigger.
  Future<void> manualCapture() async {
    if (_isCapturing || scanState.value != ScanState.scanning) return;

    final layout = activeGridLayout;
    int row = _pitchToRow(pitch.value);
    int col = _yawToCol(yaw.value, row);

    // Snap to pole when pointing near it
    if (targetRow.value == 0 && pitch.value > 50) {
      row = 0;
      col = 0;
    } else if (targetRow.value == _lastRowIndex && pitch.value < -50) {
      row = _lastRowIndex;
      col = 0;
    }

    if (row < 0 || row >= layout.length) return;
    if (col < 0 || col >= (layout[row]['count'] as int)) return;

    await _captureCurrentFrame(yaw.value, pitch.value, row, col);
  }

  // ── Stop / Complete ───────────────────────────────────────────────

  Future<void> stopSession() async {
    _orientationSub?.cancel();
    _stableTimer?.cancel();
    cameraController?.dispose();
    cameraController = null;

    try {
      await _repository.stopSession();

      final String csvPath = await _generateMetadataCsv();

      scanState.value = ScanState.completed;

      Get.offNamed(
        Routes.SCAN_RAW_RESULT,
        arguments: {
          'sessionDir': sessionDir.path,
          'metadataPath': csvPath,
          'frameCount': capturedFrames.length,
          'frames': capturedFrames.toList(),
          'coveragePercent': coveragePercent.value,
        },
      );
    } catch (e) {
      errorMessage.value = 'Stop failed: $e';
    }
  }

  /// Generates a CSV file containing all frame metadata.
  Future<String> _generateMetadataCsv() async {
    final StringBuffer csv = StringBuffer();
    csv.writeln(
      'filename,yaw,pitch,roll,focal_length,fx,fy,cx,cy,exposure,iso,aperture',
    );

    for (final frame in capturedFrames) {
      final String fileName = path.basename(frame.filePath);
      csv.writeln(
        '$fileName,'
        '${frame.yaw.toStringAsFixed(4)},'
        '${frame.pitch.toStringAsFixed(4)},'
        '${frame.roll.toStringAsFixed(4)},'
        '${frame.focalLength.toStringAsFixed(2)},'
        '${frame.fx.toStringAsFixed(2)},'
        '${frame.fy.toStringAsFixed(2)},'
        '${frame.cx.toStringAsFixed(2)},'
        '${frame.cy.toStringAsFixed(2)},'
        '${frame.exposureTime.toStringAsFixed(6)},'
        '${frame.iso.toStringAsFixed(0)},'
        '${frame.aperture.toStringAsFixed(2)}',
      );
    }

    final String csvPath = '${sessionDir.path}/metadata.csv';
    final File csvFile = File(csvPath);
    await csvFile.writeAsString(csv.toString());
    return csvPath;
  }

  /// Cancel without saving.
  Future<void> cancelSession() async {
    _orientationSub?.cancel();
    _stableTimer?.cancel();
    cameraController?.dispose();
    cameraController = null;

    try {
      await _repository.stopSession();
    } catch (_) {}

    Get.back();
  }

  // ── Grid helpers ──────────────────────────────────────────────────

  void _updateNextTarget() {
    final layout = activeGridLayout;
    final int rowCount = layout.length;

    // Scanning order: Horizon → Top → Bottom → Zenith → Nadir
    const rowOrder = [2, 1, 3, 0, 4];

    for (final r in rowOrder) {
      if (r >= rowCount) continue;
      final int count = layout[r]['count'] as int;
      for (int c = 0; c < count; c++) {
        if (!sphereGrid[r][c]) {
          targetRow.value = r;
          targetCol.value = c;
          return;
        }
      }
    }
  }

  int _yawToCol(double y, int row) {
    final layout = activeGridLayout;
    if (row < 0 || row >= layout.length) return 0;
    final int count = layout[row]['count'] as int;
    final double offset = layout[row]['yawOffset'] as double;
    if (count <= 1) return 0;

    double normalizedYaw = (y - offset) % 360;
    if (normalizedYaw < 0) normalizedYaw += 360;

    return (normalizedYaw / (360.0 / count)).floor().clamp(0, count - 1);
  }

  /// Maps a pitch angle to the nearest grid row index.
  ///
  /// Both grids are 5 rows but have different pitch centers,
  /// so the boundary thresholds differ.
  int _pitchToRow(double p) {
    if (isWideAngle.value) {
      // Wide-angle: pitches at 90, 45, 0, -45, -90
      if (p > 70) return 0;       // Zenith
      if (p > 22.5) return 1;     // +45°
      if (p > -22.5) return 2;    // Horizon
      if (p > -70) return 3;      // -45°
      return 4;                   // Nadir
    }

    // Main camera: pitches at 90, 35, 0, -35, -90
    if (p > 70) return 0;         // Zenith
    if (p > 17.5) return 1;       // +35°
    if (p > -17.5) return 2;      // Horizon
    if (p > -70) return 3;        // -35°
    return 4;                     // Nadir
  }

  void _updateCoverage() {
    int captured = 0;
    int total = 0;
    for (int r = 0; r < sphereGrid.length; r++) {
      final row = sphereGrid[r];
      total += row.length;
      for (final cell in row) {
        if (cell) captured++;
      }
    }
    coveragePercent.value = total > 0 ? captured / total : 0;
  }

  int get currentCol => _yawToCol(yaw.value, currentRow);
  int get currentRow => _pitchToRow(pitch.value);

  bool get canComplete {
    // All non-pole rows must be complete (rows 1..n-2), or total coverage ≥ 95%.
    bool middleComplete = true;
    for (int r = 1; r < _lastRowIndex; r++) {
      for (final cell in sphereGrid[r]) {
        if (!cell) {
          middleComplete = false;
          break;
        }
      }
      if (!middleComplete) break;
    }
    return middleComplete || coveragePercent.value >= _minCoverage;
  }
}

