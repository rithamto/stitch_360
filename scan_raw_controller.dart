import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
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
/// Uses **native Camera2 API** to discover all cameras via
/// [CameraManager.getCameraIdList()] and identify wide-angle lenses
/// through [LENS_INFO_AVAILABLE_FOCAL_LENGTHS].
///
/// Supports two grid densities:
///   - **Main camera** (1x, HFOV ≈ 65°): 33-point grid across 5 rows
///   - **Wide-angle camera** (0.5x, HFOV ≈ 120°): 33-point grid across 5 rows
///
/// Auto-captures when the device is held stable and aligned with a target cell.
class ScanRawController extends GetxController {
  ScanRawController({required IPanoramaRepository repository})
    : _repository = repository;

  final IPanoramaRepository _repository;

  // ── Grid configurations ────────────────────────────────────────────

  /// Main camera grid (1x): 33 points across 5 rows.
  /// Staggered 'Diamond' pattern: Horizon has more points than Top/Bottom.
  ///
  ///   Row 0  Zenith (+90°) :  1 point
  ///   Row 1  Top    (+35°) : 10 points  (staggered in Row 2 gaps)
  ///   Row 2  Horizon ( 0°) : 11 points  (reference)
  ///   Row 3  Bottom (-35°) : 10 points  (staggered in Row 2 gaps)
  ///   Row 4  Nadir  (-90°) :  1 point
  static const List<Map<String, dynamic>> _mainCameraGrid = [
    {'pitch': 90.0,  'count': 1,  'yawOffset': 0.0},
    {'pitch': 35.0,  'count': 10, 'yawOffset': 16.36},
    {'pitch': 0.0,   'count': 11, 'yawOffset': 0.0},
    {'pitch': -35.0, 'count': 10, 'yawOffset': 16.36},
    {'pitch': -90.0, 'count': 1,  'yawOffset': 0.0},
  ];

  /// Wide-angle camera grid (0.5x): 33 points across 5 rows.
  /// Optimized for HFOV ≈ 120°.
  ///
  ///   Row 0  Zenith (+90°) :  1 point
  ///   Row 1  Top    (+45°) : 10 points  (staggered)
  ///   Row 2  Horizon ( 0°) : 11 points  (reference)
  ///   Row 3  Bottom (-45°) : 10 points  (staggered)
  ///   Row 4  Nadir  (-90°) :  1 point
  static const List<Map<String, dynamic>> _wideCameraGrid = [
    {'pitch': 90.0,  'count': 1,  'yawOffset': 0.0},
    {'pitch': 45.0,  'count': 10, 'yawOffset': 16.36},
    {'pitch': 0.0,   'count': 11, 'yawOffset': 0.0},
    {'pitch': -45.0, 'count': 10, 'yawOffset': 16.36},
    {'pitch': -90.0, 'count': 1,  'yawOffset': 0.0},
  ];

  /// Whether the wide-angle camera is currently active.
  final isWideAngle = false.obs;

  /// The currently active grid layout (reactive).
  List<Map<String, dynamic>> get activeGridLayout =>
      isWideAngle.value ? _wideCameraGrid : _mainCameraGrid;

  static const double _minCoverage = 0.95;

  // ── Native Camera State ────────────────────────────────────────────
  /// All discovered cameras from Camera2 API.
  final discoveredCameras = <CameraLensInfo>[].obs;

  /// Currently selected camera lens info.
  final selectedCamera = Rxn<CameraLensInfo>();

  /// The main (1x) back camera — longest focal length among back cameras.
  CameraLensInfo? _mainCamera;

  /// The wide-angle (0.5x) back camera — shortest focal length / widest HFOV.
  CameraLensInfo? _wideCamera;

  /// Texture ID from native Camera2 preview for Flutter [Texture] widget.
  final nativeTextureId = (-1).obs;

  /// Whether multiple back cameras are available for lens switching.
  final hasMultipleLenses = false.obs;

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
  final sphereGrid = <RxList<bool>>[].obs;

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
    _repository.closeNativeCamera();
    super.onClose();
  }

  // ── Grid management ────────────────────────────────────────────────

  /// (Re)builds the sphere grid to match the active layout.
  void _buildSphereGrid() {
    final layout = activeGridLayout;
    sphereGrid.assignAll(List.generate(
      layout.length,
      (r) => RxList.generate(layout[r]['count'] as int, (_) => false),
    ));
  }

  int _currentCamIdx = 0;

  /// Cycles through all available back-facing cameras.
  /// Uses native Camera2 API to physically switch the camera lens.
  Future<void> switchCameraLens() async {
    final backCams = discoveredCameras.where((c) => c.isBackFacing).toList();
    if (backCams.length <= 1) return;

    _currentCamIdx = (_currentCamIdx + 1) % backCams.length;
    final targetCamera = backCams[_currentCamIdx];

    try {
      final textureId = await _repository.switchNativeCamera(
        targetCamera.cameraId,
      );

      nativeTextureId.value = textureId;
      selectedCamera.value = targetCamera;
      isWideAngle.value = targetCamera.isWideAngle;

      // Reset grid for new lens density
      _buildSphereGrid();
      capturedFrames.clear();
      coveragePercent.value = 0.0;
      _updateNextTarget();

      // Show temporary feedback on current camera
      Get.snackbar(
        'Camera Switched',
        'ID: ${targetCamera.cameraId} (${targetCamera.hfov.round()}° FOV)',
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.black54,
        colorText: Colors.white,
        duration: const Duration(seconds: 2),
      );
    } catch (e) {
      Get.snackbar('Error', e.toString());
    }
  }

  // ── Initialize Native Camera + IMU ─────────────────────────────────

  Future<void> _initializeSession() async {
    try {
      scanState.value = ScanState.initializing;

      // 1. Discover all cameras via Camera2 API
      final cameras = await _repository.discoverCameras();
      discoveredCameras.assignAll(cameras);

      // Log all discovered cameras for debugging
      for (final cam in cameras) {
        debugPrint(
          '[NativeCamera] Discovered: ID=${cam.cameraId} '
          'facing=${cam.facing} '
          'focalLengths=${cam.focalLengths} '
          'HFOV=${cam.hfov.toStringAsFixed(1)}° '
          'isWideAngle=${cam.isWideAngle} '
          'maxRes=${cam.maxWidth}x${cam.maxHeight}',
        );
      }

      // 2. Identify main and wide-angle back cameras based on HFOV
      final backCameras = cameras.where((c) => c.isBackFacing).toList();
      if (backCameras.isEmpty) throw Exception('No back camera found');

      // Sort by HFOV descending (widest first), then by isPhysical (physical first)
      backCameras.sort((a, b) {
        final cmp = b.hfov.compareTo(a.hfov);
        if (cmp != 0) return cmp;
        if (a.isPhysical != b.isPhysical) return b.isPhysical ? 1 : -1;
        return 0;
      });

      // Widest is our "Wide" candidate (e.g. 0.5x)
      _wideCamera = backCameras.first;

      // Narrowest back camera is our "Main" candidate (e.g. 1x)
      _mainCamera = backCameras.lastWhere((c) => !c.isWideAngle, orElse: () => backCameras.last);

      // If they are the same, we only have one lens
      if (_wideCamera?.cameraId == _mainCamera?.cameraId) {
        _wideCamera = null;
      }

      hasMultipleLenses.value = _wideCamera != null;

      debugPrint(
        '[NativeCamera] Main camera: ${_mainCamera?.cameraId} '
        '(HFOV=${_mainCamera?.hfov.toStringAsFixed(1)}°)',
      );
      debugPrint(
        '[NativeCamera] Wide camera: ${_wideCamera?.cameraId} '
        '(HFOV=${_wideCamera?.hfov.toStringAsFixed(1)}°)',
      );

      // 3. Open the wide-angle camera by default if available, otherwise main
      final startCamera = _wideCamera ?? _mainCamera!;
      final textureId = await _repository.openNativeCamera(
        startCamera.cameraId,
      );
      nativeTextureId.value = textureId;
      selectedCamera.value = startCamera;
      isWideAngle.value = startCamera == _wideCamera;
      
      // Rebuild grid to match the starting lens density
      _buildSphereGrid();
      _updateNextTarget();

      debugPrint(
        '[NativeCamera] Opened ${startCamera.zoomLabel} camera ${startCamera.cameraId} → textureId=$textureId',
      );

      // 4. Initialize IMU
      await _repository.initialize();

      // 5. Prepare storage
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final tempDir = Directory.systemTemp;
      sessionDir = Directory('${tempDir.path}/panorama_$timestamp');
      await sessionDir.create();

      // 6. Listen to orientation stream
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

  // ── Capture logic (Native Camera2) ────────────────────────────────

  Future<void> _captureCurrentFrame(
    double captureYaw,
    double capturePitch,
    int row,
    int col,
  ) async {
    if (_isCapturing) return;
    _isCapturing = true;

    try {
      final int timestamp = DateTime.now().millisecondsSinceEpoch;

      // Build file path - simplified naming (sequence number only)
      final String destPath = '${sessionDir.path}/${capturedFrames.length}.jpg';

      // Capture via native Camera2 API
      final result = await _repository.captureNativeImage(destPath);

      final bool success = result['success'] as bool? ?? false;
      if (!success) {
        debugPrint('[NativeCamera] Capture failed: ${result['error']}');
        return;
      }

      // Extract real intrinsics from Camera2 characteristics
      final double fx = (result['fx'] as num?)?.toDouble() ?? 0.0;
      final double fy = (result['fy'] as num?)?.toDouble() ?? 0.0;
      final double cx = (result['cx'] as num?)?.toDouble() ?? 0.0;
      final double cy = (result['cy'] as num?)?.toDouble() ?? 0.0;
      final double focalLengthMm =
          (result['focalLengthMm'] as num?)?.toDouble() ?? 0.0;
      final double hfov = (result['hfov'] as num?)?.toDouble() ?? 0.0;

      final frame = RawFrame(
        index: capturedFrames.length,
        clusterId: _getClusterId(row, col),
        filePath: destPath,
        yaw: captureYaw,
        pitch: capturePitch,
        roll: roll.value,
        focalLength: focalLengthMm,
        capturedAt: DateTime.now(),
        fx: fx,
        fy: fy,
        cx: cx,
        cy: cy,
      );

      capturedFrames.add(frame);

      sphereGrid[row][col] = true;
      sphereGrid.refresh();
      _updateCoverage();
      _updateNextTarget();

      HapticFeedback.mediumImpact();

      debugPrint(
        '[NativeCamera] Captured frame ${frame.index}: '
        'yaw=${captureYaw.toStringAsFixed(1)}° '
        'pitch=${capturePitch.toStringAsFixed(1)}° '
        'fx=${fx.toStringAsFixed(1)} fy=${fy.toStringAsFixed(1)} '
        'HFOV=${hfov.toStringAsFixed(1)}° '
        'focalMm=${focalLengthMm.toStringAsFixed(2)}',
      );
    } catch (e) {
      debugPrint('[NativeCamera] Capture failed: $e');
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
    await _repository.closeNativeCamera();
    nativeTextureId.value = -1;

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
      'filename,cluster_id,yaw,pitch,roll,focal_length,fx,fy,cx,cy,exposure,iso,aperture',
    );

    for (final frame in capturedFrames) {
      final String fileName = path.basename(frame.filePath);
      csv.writeln(
        '$fileName,'
        '${frame.clusterId},'
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
    await _repository.closeNativeCamera();
    nativeTextureId.value = -1;

    try {
      await _repository.stopSession();
    } catch (_) {}

    Get.back();
  }

  // ── Grid helpers ──────────────────────────────────────────────────

  void _updateNextTarget() {
    final layout = activeGridLayout;
    final int rowCount = layout.length;

    // Scanning order: Cluster-based (Vertical Strips)
    // For each column, capture Horizon (2) -> Top (1) -> Bottom (3)
    final int colCountInRow2 = layout[2]['count'] as int;

    for (int j = 0; j < colCountInRow2; j++) {
      const rowOrder = [2, 1, 3];
      for (final r in rowOrder) {
        if (r >= rowCount) continue;
        final int count = layout[r]['count'] as int;
        if (j < count && !sphereGrid[r][j]) {
          targetRow.value = r;
          targetCol.value = j;
          return;
        }
      }
    }

    // Finally Zenith (0) and Nadir (4)
    if (!sphereGrid[0][0]) {
      targetRow.value = 0;
      targetCol.value = 0;
      return;
    }
    if (rowCount > 4 && !sphereGrid[4][0]) {
      targetRow.value = 4;
      targetCol.value = 0;
      return;
    }
  }

  /// Calculates a unique cluster ID for grouping frames.
  /// Middle rows are grouped by column index (0-10).
  /// Zenith and Nadir are their own clusters.
  int _getClusterId(int row, int col) {
    if (row == 0) return 100; // Zenith cluster
    if (row == 4) return 200; // Nadir cluster
    return col; // Middle rows grouped by column
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
