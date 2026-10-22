const test = require("node:test"),
  assert = require("node:assert/strict"),
  S = require("../frontend/static/spatial-export.js");
test("constant region includes every camera correction and transforms reversibly", () => {
  const p = {
    wells: [
      {
        points: [
          [0.2, 0.2],
          [0.6, 0.2],
          [0.6, 0.6],
          [0.2, 0.6],
        ],
        overrides: {
          10: [
            [0.3, 0.3],
            [0.8, 0.3],
            [0.8, 0.7],
            [0.3, 0.7],
          ],
        },
      },
    ],
  };
  const original = JSON.stringify(p),
    r = S.bounds(p, 1000, 800, 32);
  assert.deepEqual(r, [0.168, 0.16, 0.664, 0.58]);
  for (const points of [p.wells[0].points, p.wells[0].overrides[10]]) {
    const q = S.transform(points, r);
    q.forEach(([x, y], i) => {
      assert.ok(x >= 0 && x <= 1 && y >= 0 && y <= 1);
      assert.ok(Math.abs(x * r[2] + r[0] - points[i][0]) < 1e-12);
    });
  }
  assert.equal(JSON.stringify(p), original);
  assert.throws(() => S.bounds(p, 1000, 800, -1), /padding/);
});
test("many corrections do not overflow the call stack; edge regions remain in image", () => {
  const points = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ],
    overrides = Object.fromEntries(
      Array.from({ length: 10000 }, (_, i) => [i, points]),
    );
  assert.deepEqual(
    S.bounds({ wells: [{ points, overrides }] }, 300, 200, 32),
    [0, 0, 1, 1],
  );
});
