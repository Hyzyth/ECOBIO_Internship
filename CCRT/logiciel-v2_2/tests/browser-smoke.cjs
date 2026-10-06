const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os");
const { execFileSync } = require("node:child_process");
const work = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-browser-"));
execFileSync(process.env.PYTHON || "python", [
  "-c",
  `
import cv2,numpy as np,sys
from pathlib import Path
p=Path(sys.argv[1]); (p/'fixtures').mkdir()
for i in [1,2,10]:
 image=np.zeros((500,700,3),np.uint8)
 cv2.rectangle(image,(50,70),(200,230),(255,255,255),4)
 cv2.ellipse(image,(420,260),(90,60),20,0,360,(255,255,255),4)
 cv2.imwrite(str(p/'fixtures'/f'frame{i}.png'),image)
`,
  work,
]);
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto(process.env.APP_URL || "http://127.0.0.1:5000");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length > 0,
  );
  await page
    .locator("#files")
    .setInputFiles([10, 2, 1].map((i) => `${work}/fixtures/frame${i}.png`));
  await page.waitForFunction(
    () => document.querySelector("#frameName").textContent === "frame1.png",
  );
  await page.locator("#detect").click();
  await page.waitForFunction(
    () => document.querySelector("#wellSelect").options.length === 3,
  );
  await page.locator("#confirm").click();
  await page.locator("#correction").selectOption("coma");
  assert.equal(await page.locator("#eggCount").isVisible(), false);
  await page.locator("#notes").fill("Observed manually");
  await page.locator("#saveCorrection").click();
  assert.match(await page.locator("#prediction").textContent(), /COMA/);
  await page.locator("#next").click();
  await page.waitForFunction(
    () => document.querySelector("#frameName").textContent === "frame2.png",
  );
  await page.locator("#correction").selectOption("awake");
  await page.locator("#saveCorrection").click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#json").click();
  const download = await downloadPromise;
  await download.saveAs(path.join(work, "project.json"));
  const p = require(path.join(work, "project.json"));
  assert.equal(p.results.length, 6);
  assert.equal(
    p.summary.find((s) => s.well_id === p.wells.at(-1).id).transitions,
    1,
  );
  assert.equal(p.results[1].prediction, "unknown");
  assert.equal(p.results[1].user_correction.egg_count, null);
  await page.locator("#wellName").fill("A3");
  await page.locator("#wellName").press("Tab");
  await page.locator("#tool").selectOption("vertex");
  const box = await page.locator("#canvas").boundingBox();
  const point = p.wells.at(-1).points[0];
  await page.mouse.move(
    box.x + point[0] * box.width,
    box.y + point[1] * box.height,
  );
  await page.mouse.down();
  await page.mouse.move(
    box.x + point[0] * box.width + 8,
    box.y + point[1] * box.height + 8,
  );
  await page.mouse.up();
  await page.locator("#import").setInputFiles(path.join(work, "project.json"));
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Project restored"),
  );
  await page.locator("#play").click();
  await page.waitForFunction(
    () => document.querySelector("#frameName").textContent === "frame10.png",
  );
  await page.waitForFunction(
    () => document.querySelector("#play").textContent === "Play",
  );
  await page.screenshot({
    path: path.join(work, "review.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: path.join(work, "mobile.png"),
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
    true,
  );
  await page.locator("#mode").selectOption("single");
  await page.locator("#layout").selectOption("individual");
  await page
    .locator("#files")
    .setInputFiles(path.join(work, "fixtures/frame2.png"));
  await page.waitForFunction(
    () => document.querySelector("#frameCounter").textContent === "1 / 1",
  );
  assert.equal(await page.locator("#play").isDisabled(), true);
  await page.waitForFunction(() => !busy);
  await page.locator("#whole").click();
  assert.equal(await page.locator("#wellSelect option").count(), 2);
  await page.locator("#correction").selectOption("awake");
  await page.locator("#saveCorrection").click();
  assert.match(await page.locator("#prediction").textContent(), /AWAKE/);
  await page.locator("#tool").selectOption("polygon");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => image !== null);
  const singleBox = await page.locator("#canvas").boundingBox();
  for (const [x, y] of [
    [0.2, 0.2],
    [0.4, 0.2],
    [0.4, 0.4],
  ])
    await page.mouse.click(
      singleBox.x + x * singleBox.width,
      singleBox.y + y * singleBox.height,
    );
  await page.mouse.dblclick(
    singleBox.x + 0.2 * singleBox.width,
    singleBox.y + 0.4 * singleBox.height,
  );
  assert.equal(await page.locator("#wellSelect option").count(), 3);
  await page.locator("#removeSelected").click();
  assert.equal(await page.locator("#proposalSelect option").count(), 2);
  assert.equal(await page.locator("#summary tr").count(), 1);
  const remaining = await page
    .locator("#proposalSelect option")
    .nth(1)
    .getAttribute("value");
  await page.locator("#proposalSelect").selectOption(remaining);
  assert.match(await page.locator("#prediction").textContent(), /AWAKE/);
  await page.locator("#canvas").click({ position: { x: 5, y: 5 } });
  // Select by ID so deletion works even when overlapping boundaries are hard to click.
  await page.locator("#proposalSelect").selectOption(remaining);
  await page.locator("#canvas").focus();
  await page.keyboard.press("Delete");
  assert.equal(await page.locator("#proposalSelect option").count(), 1);
  assert.equal(await page.locator("#summary tr").count(), 0);
  assert.deepEqual(errors, []);
  await browser.close();
  console.log("Browser artifacts:", work);
  console.log(
    "PASS browser: load/order, detect, annotate, preview/navigation, geometry edits, export/restore, playback, responsive layout, toolbar/keyboard deletion; no JS errors",
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
