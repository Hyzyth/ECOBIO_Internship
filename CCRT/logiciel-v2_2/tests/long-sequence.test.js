const test = require("node:test"),
  assert = require("node:assert/strict"),
  C = require("../frontend/static/review-core.js"),
  T = require("../frontend/static/training-export.js");

test("full-sequence runs retain late recovery and isolated uncertain frames", () => {
  const p = {
      frames: Array.from({ length: 5400 }, () => ({})),
      mode: "sequence",
      analysis_start_index: 100,
      wells: [],
      records: {},
    },
    w = { uid: "one", occupancy: "single" };
  for (let i = 100; i < 5400; i++)
    p.records[C.key(i, "one")] = {
      correction: { state: i < 5100 ? "coma" : "awake" },
    };
  p.records[C.key(5099, "one")].correction.state = "unknown";
  assert.deepEqual(C.stateRuns(p, w), [
    { state: "coma", start: 100, end: 5098 },
    { state: "unknown", start: 5099, end: 5099 },
    { state: "awake", start: 5100, end: 5399 },
  ]);
  w.occupancy = "empty";
  assert.deepEqual(C.stateRuns(p, w), [
    { state: "empty", start: 100, end: 5399 },
  ]);
});

test("resizing preserves geometry and clips edge wells without crossing edges", () => {
  const points = [
      [0.2, 0.2],
      [0.4, 0.2],
      [0.4, 0.4],
      [0.2, 0.4],
    ],
    saved = structuredClone(points);
  const r = C.resizeBoundary(points, 1.1);
  assert.ok(r.points[0][0] < 0.2);
  assert.deepEqual(points, saved);
  assert.equal(r.partial, false);
  const clipped = C.resizeBoundary(
    [
      [0, 0.2],
      [0.1, 0.2],
      [0.1, 0.4],
      [0, 0.4],
    ],
    1.1,
  );
  assert.ok(C.isSimplePolygon(clipped.points));
  assert.ok(clipped.partial);
  assert.equal(Math.min(...clipped.points.map((p) => p[0])), 0);
  assert.throws(() => C.resizeBoundary(points, 2), /20%/);
});

test("ZIP parts are independently readable, bounded and cancellable", async () => {
  const entries = Array.from({ length: 7 }, (_, i) => ({
    name: `images/f${i}.bin`,
    blob: new Blob([new Uint8Array(128).fill(i)]),
  }));
  const progress = [];
  const parts = await T.zipParts(
    entries,
    (n, total) => progress.push([n, total]),
    () => false,
    500,
  );
  assert.ok(parts.length > 1);
  assert.ok(parts.every((p) => p.size <= 500));
  let names = [];
  for (const part of parts) {
    const bytes = new Uint8Array(await part.arrayBuffer()),
      dv = new DataView(bytes.buffer);
    let pos = 0;
    while (dv.getUint32(pos, true) === 0x04034b50) {
      const size = dv.getUint32(pos + 18, true),
        len = dv.getUint16(pos + 26, true),
        start = pos + 30 + len;
      names.push(new TextDecoder().decode(bytes.slice(pos + 30, start)));
      assert.equal(
        T.crc32(bytes.slice(start, start + size)),
        dv.getUint32(pos + 14, true),
      );
      pos = start + size;
    }
    assert.equal(dv.getUint32(pos, true), 0x02014b50);
  }
  assert.deepEqual(
    names,
    entries.map((e) => e.name),
  );
  assert.deepEqual(progress.at(-1), [7, 7]);
  await assert.rejects(
    () =>
      T.zipParts(
        entries,
        () => {},
        () => true,
        500,
      ),
    /cancelled/,
  );
});

test("a failed disk write aborts the file and keeps the original error", async () => {
  let aborted = false;
  const directory = {
    getFileHandle: async () => ({
      createWritable: async () => ({
        write: async () => {
          throw Error("Disk full");
        },
        abort: async () => {
          aborted = true;
        },
        close: async () => assert.fail("must not close a failed write"),
      }),
    }),
  };
  await assert.rejects(
    () =>
      T.writeEntry(
        directory,
        { name: "annotations.json", text: "{}" },
        () => false,
      ),
    /Disk full/,
  );
  assert.ok(aborted);
});
