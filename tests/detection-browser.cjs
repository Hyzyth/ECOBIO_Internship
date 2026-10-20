/* Test the exact app/API path; optional private images remain outside Git. */
const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process"),
  { step } = require("./workflow-helpers.cjs");
(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-rims-")),
    python = process.env.PYTHON || "python";
  let file = process.env.DETECTION_IMAGE;
  if (!file) {
    file = path.join(out, "inner-reflections.png");
    execFileSync(python, [
      "-c",
      `import cv2
from tests.test_detection_variations import inner_reflection_scene
image,_=inner_reflection_scene()
cv2.imwrite(${JSON.stringify(file)},image)`,
    ]);
  }
  const browser = await chromium.launch({
      executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
      headless: true,
      args: ["--no-sandbox"],
    }),
    page = await browser.newPage({ viewport: { width: 1440, height: 1100 } }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page.locator("#files").setInputFiles(file);
  await page.waitForFunction(() => image !== null);
  await step(page, "wells");
  await page.locator("#detect").click();
  await page.waitForFunction(() => !busy && project.wells.length > 0);
  const saved = await page.evaluate(() => C.projectBackup(project)),
    expected = Number(process.env.DETECTION_EXPECTED_COUNT || 61);
  assert.equal(saved.wells.length, expected);
  assert.equal(
    saved.detection_settings.algorithm_version,
    "native-outer-rims-v3",
  );
  if (!process.env.DETECTION_IMAGE)
    assert.ok(saved.wells.some((w) => w.detection.recovered_from_neighbours));
  assert.ok(
    saved.wells.every(
      (w) => w.detection.algorithm_version === "native-outer-rims-v3",
    ),
  );
  assert.ok(saved.wells.every((w) => COVERAGE(w)));
  function COVERAGE(w) {
    return (
      w.detection.outer_rim_coverage >= 0 && w.detection.outer_rim_coverage <= 1
    );
  }
  fs.writeFileSync(path.join(out, "project.json"), JSON.stringify(saved));
  await page.screenshot({
    path: path.join(out, "detected.png"),
    fullPage: true,
  });
  // A browser reload without a Flask restart previously silently ran old code.
  await page.evaluate(() => {
    project.records[C.key(0, project.wells[0].uid)] = {
      correction: { state: "coma" },
    };
  });
  const before = await page.evaluate(() =>
    JSON.stringify({ wells: project.wells, records: project.records }),
  );
  await page.route("**/api/detect", (route) =>
    route.fulfill({ json: { wells: [], settings: {}, warning: "old server" } }),
  );
  await page.locator("#detect").click();
  await page.waitForFunction(() => !busy);
  assert.match(
    await page.locator("#status").textContent(),
    /older detector.*Restart the app/,
  );
  assert.equal(
    await page.evaluate(() =>
      JSON.stringify({ wells: project.wells, records: project.records }),
    ),
    before,
  );
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    `PASS app default detection: ${expected} wells, outer-rim provenance and stale-server protection. Artifacts: ${out}`,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
