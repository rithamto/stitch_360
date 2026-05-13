import csv, re

frames = []
with open('drive-download-20260513T035900Z-3-001/metadata.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        m = re.search(r'frame_(\d+)', fname)
        fid = int(m.group(1))
        yaw = float(row['yaw'])
        pitch = float(row['pitch'])
        roll = float(row['roll'])
        if pitch > 70: rowname = 'Zenith'
        elif pitch > 17.5: rowname = 'Top'
        elif pitch >= -17.5: rowname = 'Horizon'
        elif pitch >= -70: rowname = 'Bottom'
        else: rowname = 'Nadir'
        cid = fid % 14 if fid < 42 else (100 if fid == 42 else 200)
        frames.append((fid, yaw, pitch, roll, rowname, cid))

frames.sort(key=lambda x: x[0])

# Group by cluster
clusters = {}
for f in frames:
    cid = f[5]
    if cid not in clusters:
        clusters[cid] = []
    clusters[cid].append(f)

print("=== Current clustering (fid % 14) ===")
for cid in sorted(clusters.keys()):
    print(f'\nCluster {cid}:')
    for f in clusters[cid]:
        print(f'  Frame {f[0]:2d}: yaw={f[1]:7.2f}, pitch={f[2]:7.2f}, roll={f[3]:7.2f}, row={f[4]}')

# The actual capture pattern from the CSV order
print("\n\n=== Actual capture sequence ===")
print("Frame | Yaw     | Pitch   | Roll    | Row")
print("-" * 55)
for f in frames:
    print(f"  {f[0]:2d}  | {f[1]:7.2f} | {f[2]:7.2f} | {f[3]:7.2f} | {f[4]}")

# Correct clustering: group by yaw proximity (vertical columns)
# The pattern is: Zenith, then groups of 3 (Top, Horizon, Bottom) per column
print("\n\n=== Proposed clustering by yaw proximity ===")
normal_frames = [f for f in frames if f[4] not in ('Zenith', 'Nadir')]
# Sort by yaw
normal_frames_sorted = sorted(normal_frames, key=lambda x: x[1])
print("\nNormal frames sorted by yaw:")
for f in normal_frames_sorted:
    print(f"  Frame {f[0]:2d}: yaw={f[1]:7.2f}, pitch={f[2]:7.2f}, row={f[4]}")

# Check the actual sequence: frames 1-41 (excluding 0=zenith, 43=nadir)
# Pattern should be: for each column, Top->Horizon->Bottom
print("\n\n=== Sequential grouping (every 3 frames) ===")
# Skip frame 0 (zenith) and frame 43 (nadir)
seq_frames = [f for f in frames if f[4] not in ('Zenith', 'Nadir')]
seq_frames.sort(key=lambda x: x[0])
for i in range(0, len(seq_frames), 3):
    group = seq_frames[i:i+3]
    yaws = [g[1] for g in group]
    avg_yaw = sum(yaws) / len(yaws)
    print(f"\nColumn {i//3} (avg yaw={avg_yaw:.1f}):")
    for g in group:
        print(f"  Frame {g[0]:2d}: yaw={g[1]:7.2f}, pitch={g[2]:7.2f}, roll={g[3]:7.2f}, row={g[4]}")
    # Check yaw spread within column
    yaw_spread = max(yaws) - min(yaws)
    print(f"  Yaw spread: {yaw_spread:.2f} degrees")
