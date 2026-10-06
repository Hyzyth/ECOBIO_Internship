const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process");
(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-restore-")),
    python = process.env.PYTHON || "python";
  execFileSync(python, [
    "-c",
    `from PIL import Image
from pathlib import Path
p=Path(${JSON.stringify(out)})
for i in range(3): Image.new('RGB',(600,400),(40+i*30,90,140)).save(p/f'frame{i}.png')`,
  ]);
  const browser = await chromium.launch({
      executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
      headless: true,
      args: ["--no-sandbox"],
    }),
    page = await browser.newPage(),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length > 0,
  );
  await page
    .locator("#files")
    .setInputFiles([0, 1, 2].map((i) => path.join(out, `frame${i}.png`)));
  await page.waitForFunction(() => image !== null);
  await page.locator("#whole").click();
  assert.equal(
    await page.locator("#boundaryApproval").inputValue(),
    "sequence",
  );
  await page.locator("#confirm").click();
  assert.equal(
    await page.evaluate(() => project.wells[0].review_frames.length),
    0,
  );
  await page.locator("#rangeStart").fill("1");
  await page.locator("#rangeEnd").fill("2");
  await page.locator("#applyRange").click();
  await page.locator("#timeline").evaluate((el) => {
    el.value = "2";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction(() => index === 2 && image !== null);
  await page.locator("#awakeFrom").click();
  const before = await page.evaluate(() => C.resultsReport(project));
  fs.writeFileSync(path.join(out, "old-report.json"), JSON.stringify(before));
  execFileSync(python, [
    "-c",
    `import zipfile
from pathlib import Path
p=Path(${JSON.stringify(out)})
with zipfile.ZipFile(p/'old-report.zip','w',compression=zipfile.ZIP_DEFLATED) as z:z.write(p/'old-report.json','report.json')`,
  ]);
  await page.evaluate(() => {
    window.paintCalls = [];
    const original = ctx.fillText.bind(ctx);
    ctx.fillText = (text, ...args) => {
      paintCalls.push({ text, color: ctx.fillStyle });
      original(text, ...args);
    };
  });
  await page.locator("#occupancy").selectOption("empty");
  const mark = await page.evaluate(() => paintCalls.at(-1));
  assert.equal(mark.text, "W1 E");
  assert.equal(mark.color, "#00cfff");
  assert.equal(
    await page
      .locator("#summary tr")
      .first()
      .locator("td")
      .nth(1)
      .textContent(),
    "Empty",
  );
  await page.locator("#occupancy").selectOption("multiple");
  assert.equal(await page.locator("#correction").inputValue(), "invalid");
  assert.equal(await page.evaluate(() => paintCalls.at(-1).text), "W1 !");
  await page.locator("#occupancy").selectOption("empty");
  assert.match(await page.locator("#prediction").textContent(), /EMPTY/);
  assert.equal(await page.locator("#correction").inputValue(), "empty");
  assert.equal(await page.locator("#labelComa").isDisabled(), true);
  assert.equal(
    await page.evaluate(() => C.summary(project)[0].empty_frames),
    3,
  );
  assert.equal(await page.evaluate(() => C.summary(project)[0].coma_frames), 0);
  const promise = page.waitForEvent("download");
  await page.locator("#report").click();
  const d = await promise;
  await d.saveAs(path.join(out, "new-report.zip"));
  await page.waitForFunction(() => !busy);
  await page.locator("#occupancy").selectOption("single");
  await page.locator("#labelAwake").click();
  await page
    .locator("#importReport")
    .setInputFiles(path.join(out, "new-report.zip"));
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Project restored"),
  );
  assert.equal(await page.locator("#occupancy").inputValue(), "empty");
  assert.deepEqual(
    await page.evaluate(() =>
      project.frames.map(
        (_, i) =>
          project.records[C.key(i, project.wells[0].uid)].correction.state,
      ),
    ),
    ["coma", "coma", "awake"],
  );
  await page.locator("#occupancy").selectOption("single");
  await page.locator("#rangeStart").fill("1");
  await page.locator("#rangeEnd").fill("3");
  await page.locator("#rangeState").selectOption("unknown");
  await page.locator("#applyRange").click();
  await page
    .locator("#importReport")
    .setInputFiles(path.join(out, "old-report.zip"));
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Imported 3 report rows"),
  );
  assert.deepEqual(await page.evaluate(() => C.summary(project)[0].states), [
    "coma",
    "coma",
    "awake",
  ]);
  await page.locator("#confirmSequence").click();
  assert.equal(
    await page.evaluate(() => project.wells[0].review_frames.length),
    0,
  );
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    "PASS restoration browser: default sequence approval, visible empty occupancy, full report ZIP round-trip with retained labels, compressed legacy report import without reannotation. Artifacts:",
    out,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
