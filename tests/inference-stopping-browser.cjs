/* Deterministic adapter responses test HTTP dispatch, not biological accuracy. */
const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process");
(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-stopping-"));
  execFileSync(process.env.PYTHON || "python", [
    "-c",
    "from tests.suite_fixtures import grids; import sys; grids(sys.argv[1])",
    root,
  ]);
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage(),
    errors = [],
    batches = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  const model = {
    id: "dispatch-test",
    available: true,
    name: "Test adapter",
    task: "ccrt",
    version: "test",
    recovery_policy: {
      confidence_threshold: 0.8,
      awake_consecutive_frames: 2,
      require_coma_before_awake: true,
    },
  };
  await page.route("**/api/models", (route) =>
    route.fulfill({
      json: [
        { id: "manual", name: "Manual", version: "1", available: true },
        model,
      ],
    }),
  );
  await page.route("**/api/analyze", async (route) => {
    const body = route.request().postDataBuffer().toString();
    const frame = Number(body.match(/name="frame_index"\r\n\r\n(\d+)/)[1]);
    const wells = JSON.parse(body.match(/name="wells"\r\n\r\n([^\r]+)/)[1]);
    batches.push(wells.map((w) => w.id));
    await route.fulfill({
      json: {
        model,
        records: wells.map((w) => ({
          well_id: w.id,
          prediction: frame >= 1 ? "awake" : "coma",
          confidence: 0.99,
        })),
      },
    });
  });
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page
    .locator("#files")
    .setInputFiles(
      Array.from({ length: 6 }, (_, i) =>
        path.join(
          root,
          "train/images",
          `frame-${String(i + 1).padStart(6, "0")}.png`,
        ),
      ),
    );
  await page.waitForFunction(() => image !== null && !busy);
  await page.evaluate(() => {
    addWell([
      [0.05, 0.05],
      [0.3, 0.05],
      [0.3, 0.4],
      [0.05, 0.4],
    ]);
    project.wells[0].detection.review_required = false;
    project.wells[0].review_frames = [];
    project.wells[0].occupancy = "single";
    $("model").value = "dispatch-test";
    $("model").onchange();
    project.recovery_policy = selectedRecoveryPolicy();
    goToStep("review");
    $("modelOptions").open = true;
  });
  // Invoke the button's handler even if the optional panel is collapsed.
  await page.evaluate(() => $("analyze").onclick());
  assert.equal(batches.length, 3, await page.locator("#status").textContent());
  const result = await page.evaluate(() => ({
    uid: project.wells[0].uid,
    records: project.records,
  }));
  assert.equal(result.records[`5:${result.uid}`].prediction, "awake");
  assert.equal(result.records[`5:${result.uid}`].raw_prediction, "unknown");
  assert.equal(
    result.records[`5:${result.uid}`].inference_skipped,
    "confirmed_awake",
  );
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    "PASS browser: 3 inference requests for 6 frames, confirmed recovery carried forward without fabricated raw outputs",
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
