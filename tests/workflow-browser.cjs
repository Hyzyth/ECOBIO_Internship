const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process"),
  { step, openOptions } = require("./workflow-helpers.cjs");

(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-workflow-"));
  execFileSync(process.env.PYTHON || "python", [
    "-c",
    `
from pathlib import Path
from PIL import Image
p=Path(${JSON.stringify(out)})
for i in range(2): Image.new('RGB',(600,400),(80+i*30,90,100)).save(p/f'frame{i}.png')
`,
  ]);
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  const base = process.env.APP_URL || "http://127.0.0.1:5000";
  const files = [0, 1].map((i) => path.join(out, `frame${i}.png`));
  await page.goto(base + "/ccrt");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length > 0,
  );
  for (const id of ["stepWells", "stepReview", "stepExport"])
    assert.equal(await page.locator("#" + id).isDisabled(), true);
  for (const id of [
    "layout",
    "interval",
    "occupancy",
    "model",
    "report",
    "importReportButton",
    "imagePanel",
    "summaryPanel",
  ])
    assert.equal(await page.locator("#" + id).isVisible(), false, id);
  await page.screenshot({ path: path.join(out, "start.png"), fullPage: true });
  await page.locator("#files").setInputFiles(files);
  await page.waitForFunction(() => image !== null);
  assert.equal(
    await page.locator("body").getAttribute("data-workflow"),
    "images",
  );
  assert.equal(await page.locator("#interval").isVisible(), true);
  assert.equal(await page.locator("#layout").isVisible(), false);
  assert.equal(await page.locator("#order").isDisabled(), false);
  await openOptions(page, "loadingOptions");
  const initialOrder = await page.evaluate(() =>
    project.frames.map((f) => f.id),
  );
  await page.locator("#order").selectOption("modified");
  assert.deepEqual(
    await page.evaluate(() => project.frames.map((f) => f.id)),
    initialOrder,
  );
  assert.equal(await page.evaluate(() => project.order), "name");
  await page.locator("#order").selectOption("name");
  await page.locator("#interval").fill("2");
  await page.locator("#interval").press("Tab");
  await page.locator("#inputContinue").click();
  assert.equal(
    await page.locator("body").getAttribute("data-workflow"),
    "wells",
  );
  assert.equal(await page.locator("#wellsContinue").isDisabled(), true);
  assert.equal(await page.locator("#occupancy").isVisible(), false);
  assert.equal(await page.locator("#detectionMethod").isVisible(), false);
  await page.locator("#drawWell").click();
  const box = await page.locator("#canvas").boundingBox();
  for (const [x, y] of [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
  ])
    await page.mouse.click(box.x + x * box.width, box.y + y * box.height);
  await page.mouse.dblclick(box.x + 0.2 * box.width, box.y + 0.8 * box.height);
  await page.waitForFunction(() => project.wells.length === 1);
  await page.locator("#confirmSequence").click();
  await page.locator("#wellsContinue").click();
  assert.equal(await page.locator("#labelComa").isVisible(), true);
  assert.equal(await page.locator("#correction").isVisible(), false);
  assert.equal(await page.locator("#analyze").isVisible(), false);
  await page.locator("#occupancy").selectOption("single");
  assert.equal(await page.locator("#confirmSingles").isVisible(), false);
  await page.locator("#labelComa").click();
  await page.locator("#next").click();
  await page.waitForFunction(() => index === 1 && image !== null);
  await page.locator("#awakeFrom").click();
  assert.equal(await page.locator("#nextLabel").isVisible(), false);
  await page.evaluate(() => {
    window.snapshot = () =>
      JSON.stringify({
        frames: project.frames,
        wells: project.wells,
        records: project.records,
        history: project.history,
        mode: project.mode,
        interval: project.interval,
        task: project.task,
        layout: project.layout,
        experiment_id: project.experiment_id,
        time_origin: project.time_origin,
      });
  });
  const saved = await page.evaluate(() => snapshot());
  // Review selection cannot change geometry or invoke the editing shortcut.
  const geometry = await page.evaluate(() => JSON.stringify(project.wells[0]));
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const reviewBox = await page.locator("#canvas").boundingBox();
  await page.mouse.move(
    reviewBox.x + 0.4 * reviewBox.width,
    reviewBox.y + 0.4 * reviewBox.height,
  );
  await page.mouse.down();
  await page.mouse.move(
    reviewBox.x + 0.45 * reviewBox.width,
    reviewBox.y + 0.45 * reviewBox.height,
  );
  await page.mouse.up();
  await page.locator("#canvas").focus();
  await page.keyboard.press("Delete");
  assert.equal(
    await page.evaluate(() => JSON.stringify(project.wells[0])),
    geometry,
  );
  for (const name of [
    "images",
    "wells",
    "review",
    "export",
    "images",
    "review",
  ]) {
    await step(page, name);
    assert.equal(await page.evaluate(() => snapshot()), saved);
    assert.equal(
      await page.locator(`[data-step="${name}"]`).getAttribute("aria-current"),
      "step",
    );
  }
  await page.screenshot({ path: path.join(out, "review.png"), fullPage: true });
  await page.locator("#reviewContinue").click();
  assert.equal(await page.locator("#summaryPanel").isVisible(), true);
  assert.ok(
    (await page.locator("#summaryPanel").boundingBox()).y <
      (await page.locator("#imagePanel").boundingBox()).y,
  );
  assert.equal(await page.locator("#comparisonSeed").isVisible(), false);
  assert.equal(await page.locator("#machine").isVisible(), false);
  await openOptions(page, "trainingOptions");
  await page.locator("#datasetTest").check();
  assert.equal(await page.locator("#comparisonOptions").isVisible(), false);
  assert.match(await page.locator("#datasetHint").textContent(), /Evaluation/);
  await page.screenshot({ path: path.join(out, "export.png"), fullPage: true });
  const download = page.waitForEvent("download");
  await page.locator("#json").click();
  await (await download).saveAs(path.join(out, "project.json"));
  await step(page, "images");
  await openOptions(page, "restoreOptions");
  await page.locator("#import").setInputFiles(path.join(out, "project.json"));
  await page.waitForFunction(() => document.body.dataset.workflow === "review");
  assert.equal(await page.evaluate(() => snapshot()), saved);
  // Busy operations keep their progress/cancel controls visible and lock navigation.
  let release;
  await page.route("**/api/detect", async (route) => {
    await new Promise((resolve) => (release = resolve));
    await route.fulfill({
      json: {
        wells: [
          {
            points: [
              [0.1, 0.1],
              [0.9, 0.1],
              [0.9, 0.9],
              [0.1, 0.9],
            ],
            source: "test",
            confidence: 0.9,
          },
        ],
        warning: "test",
        settings: {},
      },
    });
  });
  await step(page, "wells");
  await page.locator("#detect").click();
  await page.waitForFunction(() => busy);
  assert.equal(await page.locator("#cancel").isVisible(), true);
  for (const id of ["stepImages", "stepWells", "stepReview", "stepExport"])
    assert.equal(await page.locator("#" + id).isDisabled(), true);
  await page.locator("#cancel").click();
  release();
  await page.waitForFunction(() => !busy);
  assert.equal(await page.evaluate(() => snapshot()), saved);
  await page.unroute("**/api/detect");
  // A cancelled mode change preserves the review; an accepted change resets the gates.
  page.removeAllListeners("dialog");
  page.once("dialog", (d) => d.dismiss());
  await step(page, "images");
  await page.locator("#mode").selectOption("single");
  assert.equal(await page.locator("#mode").inputValue(), "sequence");
  assert.equal(await page.evaluate(() => snapshot()), saved);
  page.once("dialog", (d) => d.accept());
  await page.locator("#mode").selectOption("single");
  assert.equal(await page.locator("#stepWells").isDisabled(), true);
  assert.equal(await page.locator("#interval").isVisible(), false);
  await page.locator("#files").setInputFiles(files[0]);
  await page.waitForFunction(() => image !== null);
  assert.equal(await page.locator("#interval").isVisible(), false);
  assert.equal(await page.locator("#timeConfirmed").isVisible(), false);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#inputContinue").click();
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
  );
  await page.screenshot({ path: path.join(out, "mobile.png"), fullPage: true });
  await page.goto(base + "/eggs");
  await page.locator("#files").setInputFiles(files);
  await page.waitForFunction(
    () => image !== null && project.wells.length === 2,
  );
  assert.equal(await page.locator("#stepWells").isVisible(), false);
  await page.locator("#inputContinue").click();
  assert.equal(
    await page.locator("body").getAttribute("data-workflow"),
    "review",
  );
  assert.equal(await page.locator("#eggCount").isVisible(), true);
  assert.equal(await page.locator("#modelOptions").isVisible(), false);
  assert.equal(await page.locator("#occupancy").isVisible(), false);
  await page.locator("#eggCount").fill("0");
  await page.locator("#saveCorrection").click();
  await step(page, "images");
  await page.locator("#eggScope").selectOption("areas");
  await page.locator("#inputContinue").click();
  assert.equal(
    await page.locator("body").getAttribute("data-workflow"),
    "wells",
  );
  assert.equal(await page.locator("#layout").isVisible(), false);
  await step(page, "review");
  assert.equal(await page.locator("#eggCount").inputValue(), "0");
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    "PASS guided workflow: prerequisite gates, contextual controls, annotations retained on navigation/import, review geometry protection, cancellable detection, optional datasets, mobile and both egg paths. Artifacts:",
    out,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
