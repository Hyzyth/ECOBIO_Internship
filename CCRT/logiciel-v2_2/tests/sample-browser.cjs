/* Run against a started Flask app with SAMPLE_DIR pointing to the extracted
   user samples. Images stay outside the repository; this does not train a model. */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os");
(async () => {
  const root = process.env.SAMPLE_DIR;
  if (!root) throw Error("Set SAMPLE_DIR to the extracted samples directory.");
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-samples-"));
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
  await page.goto(process.env.APP_URL || "http://127.0.0.1:5000");
  const files = fs
    .readdirSync(path.join(root, "sequence"))
    .filter((n) => n.endsWith(".jpg"))
    .sort()
    .map((n) => path.join(root, "sequence", n));
  assert.equal(
    files.length,
    5,
    "This regression checks the supplied five-frame extract.",
  );
  await page.locator("#files").setInputFiles(files);
  await page.waitForFunction(() => image !== null);
  await page.locator("#detect").click();
  await page.waitForFunction(
    () => !busy && document.querySelector("#wellSelect").options.length === 79,
  );
  const ids = await page
    .locator("#proposalSelect option")
    .evaluateAll((options) => options.slice(1).map((o) => o.value));
  await page.locator("#tool").selectOption("region");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const box = await page.locator("#canvas").boundingBox();
  await page.mouse.move(box.x + 0.15 * box.width, box.y + 0.21 * box.height);
  await page.mouse.down();
  await page.mouse.move(box.x + 0.82 * box.width, box.y + 0.965 * box.height);
  await page.mouse.up();
  await page.waitForFunction(() => project.detection_region !== undefined);
  await page.locator("#detect").click();
  await page.waitForFunction(
    () => !busy && document.querySelector("#wellSelect").options.length === 73,
  );
  await page.locator("#clearRegion").click();
  await page.locator("#detect").click();
  await page.waitForFunction(() => !busy && project.wells.length === 78);
  const referenceIds = await page
    .locator("#proposalSelect option")
    .evaluateAll((options) => options.slice(1).map((o) => o.value));
  await page.locator("#alignSequence").click();
  await page.waitForFunction(
    () =>
      !busy &&
      document
        .querySelector("#status")
        .textContent.startsWith("Alignment finished"),
    null,
    { timeout: 60000 },
  );
  assert.match(
    await page.locator("#status").textContent(),
    /4 frames aligned.*0 failures/,
  );
  for (let i = 0; i < 5; i++) {
    await page.locator("#timeline").evaluate((el, value) => {
      el.value = String(value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, i);
    await page.waitForFunction((i) => index === i && image !== null, i);
    assert.deepEqual(
      await page
        .locator("#proposalSelect option")
        .evaluateAll((options) => options.slice(1).map((o) => o.value)),
      referenceIds,
    );
    await page.locator("#confirmFrame").click();
  }
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#json").click();
  const download = await downloadPromise;
  const projectPath = path.join(output, "sample-project.json");
  await download.saveAs(projectPath);
  const saved = JSON.parse(fs.readFileSync(projectPath, "utf8"));
  assert.equal(saved.frames.length, 5);
  assert.equal(saved.wells.length, 78);
  assert.equal(saved.results.length, 390);
  assert.equal(Object.keys(saved.alignments).length, 4);
  assert.ok(
    Object.values(saved.alignments).every(
      (a) => a.working_image_size[0] === 1200 && a.working_image_size[1] > 0,
    ),
  );
  assert.ok(
    saved.wells.every(
      (w) =>
        Object.keys(w.overrides).length === 4 && w.review_frames.length === 0,
    ),
  );
  assert.ok(
    saved.results.every(
      (r) => r.prediction === "unknown" && r.confidence === null,
    ),
  );
  await page.screenshot({
    path: path.join(output, "sequence-review.png"),
    fullPage: true,
  });
  await page.locator("#import").setInputFiles(projectPath);
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Project restored"),
  );
  assert.equal(await page.locator("#proposalSelect option").count(), 79);
  let alignmentCalls = 0;
  await page.route("**/api/align", async (route) => {
    alignmentCalls++;
    if (alignmentCalls === 2)
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({ error: "Test-only weak-match rejection" }),
      });
    else await route.continue();
  });
  await page.locator("#alignSequence").click();
  await page.waitForFunction(
    () =>
      !busy &&
      document
        .querySelector("#status")
        .textContent.startsWith("Alignment finished"),
    null,
    { timeout: 60000 },
  );
  assert.match(
    await page.locator("#status").textContent(),
    /3 frames aligned.*1 failures/,
  );
  assert.equal(
    await page.evaluate(() =>
      project.wells.every((w) => w.review_frames.includes(2)),
    ),
    true,
  );
  assert.equal(
    await page.evaluate(
      () =>
        project.alignments[2] === undefined && !!project.alignment_failures[2],
    ),
    true,
  );
  assert.deepEqual(
    await page
      .locator("#proposalSelect option")
      .evaluateAll((options) => options.slice(1).map((o) => o.value)),
    referenceIds,
  );
  await page.unroute("**/api/align");
  await page.locator("#mode").selectOption("single");
  await page.locator("#layout").selectOption("individual");
  for (const [folder, file] of ["coma", "awake"].flatMap((folder) =>
    fs
      .readdirSync(path.join(root, folder))
      .filter((file) => file.endsWith(".png"))
      .map((file) => [folder, file]),
  )) {
    await page.locator("#files").setInputFiles(path.join(root, folder, file));
    await page.waitForFunction(
      () =>
        image !== null &&
        document.querySelector("#frameCounter").textContent === "1 / 1" &&
        !busy &&
        project.wells.length === 1 &&
        project.wells[0].detection.source === "single-rim-proposal",
    );
    assert.equal(await page.locator("#proposalSelect option").count(), 2);
    const rim = await page.evaluate(() => project.wells[0]);
    assert.ok(
      rim.points.length > 20,
      "Rim polygon, not a full-image rectangle",
    );
    assert.ok(
      Math.max(...rim.points.map((p) => p[0])) -
        Math.min(...rim.points.map((p) => p[0])) >
        0.7,
    );
    await page.screenshot({ path: path.join(output, `${folder}-${file}.png`) });
    assert.match(await page.locator("#prediction").textContent(), /UNKNOWN/);
    await page.locator("#correction").selectOption(folder);
    await page.locator("#saveCorrection").click();
    assert.match(
      await page.locator("#prediction").textContent(),
      new RegExp(folder.toUpperCase()),
    );
    assert.match(
      await page.locator("#prediction").textContent(),
      /Confidence: Not available/,
    );
  }
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    "PASS real browser: 78 wells (6 clipped), ROI, 4 frame alignments, stable IDs, bulk frame review, 390 unknown-state records, export/restore, automatic rim detection on all 14 coma/awake crops, failed alignment preserves IDs and requires review. Artifacts:",
    output,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
