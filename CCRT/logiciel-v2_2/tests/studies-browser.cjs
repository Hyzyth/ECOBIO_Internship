/* Validate separate study pages and actual ZIP image/annotation/loader exports. */
const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process");
(async () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-studies-")),
    python = process.env.PYTHON || "python";
  execFileSync(python, [
    "-c",
    `from PIL import Image,ImageDraw
from pathlib import Path
p=Path(${JSON.stringify(output)})
for i in range(3):
 im=Image.new('RGB',(600,400),(30+i*40,80,130));ImageDraw.Draw(im).rectangle((120,80,480,320),outline='white',width=3);im.save(p/f'frame{i+1}.png')
`,
  ]);
  const browser = await chromium.launch({
      executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
      headless: true,
      args: ["--no-sandbox"],
    }),
    page = await browser.newPage({ viewport: { width: 1440, height: 1100 } }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  const base = process.env.APP_URL || "http://127.0.0.1:5000",
    files = [1, 2, 3].map((i) => path.join(output, `frame${i}.png`));
  async function frame(i) {
    await page.locator("#timeline").evaluate((el, i) => {
      el.value = String(i);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, i);
    await page.waitForFunction((i) => index === i && image !== null, i);
  }
  async function download(id, name) {
    const promise = page.waitForEvent("download", { timeout: 60000 });
    await page.locator("#" + id).click();
    const d = await promise;
    await d.saveAs(path.join(output, name));
    await page.waitForFunction(() => !busy);
  }
  await page.goto(base + "/ccrt");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length > 0,
  );
  assert.equal(await page.locator("#eggCount").isVisible(), false);
  assert.equal(await page.locator("#interval").isVisible(), true);
  await page.locator("#files").setInputFiles(files);
  await page.waitForFunction(() => image !== null);
  await page.locator("#experiment").fill("synthetic-ccrt-run");
  await page.locator("#experiment").press("Tab");
  await page.locator("#tool").selectOption("polygon");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const box = await page.locator("#canvas").boundingBox();
  for (const [x, y] of [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
  ])
    await page.mouse.click(box.x + x * box.width, box.y + y * box.height);
  await page.mouse.dblclick(box.x + 0.2 * box.width, box.y + 0.8 * box.height);
  await page.waitForFunction(() => project.wells.length === 1);
  await page.locator("#occupancy").selectOption("single");
  await page.locator("#nextBoundary").click();
  await page.waitForFunction(() => index === 1 && image !== null);
  await page.locator("#nextLabel").click();
  await page.waitForFunction(() => index === 2 && image !== null);
  await frame(1);
  await page.locator("#comaUntil").click();
  await frame(2);
  await page.locator("#awakeFrom").click();
  assert.deepEqual(await page.evaluate(() => C.summary(project)[0].states), [
    "coma",
    "coma",
    "awake",
  ]);
  await frame(1);
  await page.locator("#labelUncertain").click();
  assert.deepEqual(await page.evaluate(() => C.summary(project)[0].states), [
    "coma",
    "unknown",
    "awake",
  ]);
  await page.locator("#undoLabels").click();
  assert.deepEqual(await page.evaluate(() => C.summary(project)[0].states), [
    "coma",
    "coma",
    "awake",
  ]);
  await page.locator("#labelUncertain").click();
  await page.locator("#tool").selectOption("vertex");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const b = await page.locator("#canvas").boundingBox();
  await page.mouse.move(b.x + 0.2 * b.width, b.y + 0.2 * b.height);
  await page.mouse.down();
  await page.mouse.move(b.x + 0.23 * b.width, b.y + 0.22 * b.height);
  await page.mouse.up();
  for (let i = 0; i < 3; i++) {
    await frame(i);
    await page.locator("#confirmFrame").click();
  }
  await page.locator("#recoveryDelay").fill("5");
  await page.locator("#recoveryDelay").press("Tab");
  assert.equal(
    await page.evaluate(() => project.time_origin.offset_seconds),
    5,
  );
  await download("machine", "ccrt-machine.zip");
  await download("training", "ccrt-training.zip");
  await download("report", "ccrt-report.zip");
  await download("json", "ccrt-project.json");
  const saved = JSON.parse(
    fs.readFileSync(path.join(output, "ccrt-project.json")),
  );
  assert.equal(saved.task, "ccrt");
  assert.ok(saved.wells[0].overrides[1]);
  await page
    .locator("#import")
    .setInputFiles(path.join(output, "ccrt-project.json"));
  await page.waitForFunction(() =>
    document
      .querySelector("#status")
      .textContent.startsWith("Project restored"),
  );
  assert.equal(
    await page.locator("#experiment").inputValue(),
    "synthetic-ccrt-run",
  );
  await page.screenshot({
    path: path.join(output, "ccrt.png"),
    fullPage: true,
  });
  await page.locator("#scope").selectOption("all");
  await page.locator("#tool").selectOption("vertex");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const shared = await page.locator("#canvas").boundingBox();
  await page.mouse.move(
    shared.x + 0.2 * shared.width,
    shared.y + 0.2 * shared.height,
  );
  await page.mouse.down();
  await page.mouse.move(
    shared.x + 0.21 * shared.width,
    shared.y + 0.21 * shared.height,
  );
  await page.mouse.up();
  assert.equal(
    await page.evaluate(() => project.wells[0].review_frames.length),
    3,
  );
  assert.equal(
    await page.evaluate(
      () =>
        C.trainingAnnotations(project).filter((a) => a.eligible_for_training)
          .length,
    ),
    0,
  );
  await page.locator("#mode").selectOption("single");
  assert.equal(await page.locator("#interval").isVisible(), false);
  assert.equal(await page.locator("#order").isVisible(), false);
  assert.equal(await page.locator("#play").isVisible(), false);
  assert.equal(await page.locator("#comaUntil").isVisible(), false);
  assert.equal(await page.locator("#align").isVisible(), false);
  await page.goto(base + "/eggs");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length > 0,
  );
  assert.equal(await page.locator("#correction").isVisible(), false);
  assert.equal(await page.locator("#labelComa").isVisible(), false);
  assert.equal(await page.locator("#interval").isVisible(), false);
  assert.equal(await page.locator("#play").isVisible(), false);
  await page.locator("#files").setInputFiles(files.slice(0, 2));
  await page.waitForFunction(
    () => image !== null && project.wells.length === 2,
  );
  assert.equal(await page.locator("#geometryControls").isVisible(), false);
  await page.locator("#eggScope").selectOption("areas");
  await page.locator("#tool").selectOption("region");
  await page.locator("#canvas").scrollIntoViewIfNeeded();
  const eggbox = await page.locator("#canvas").boundingBox();
  await page.mouse.move(
    eggbox.x + 0.1 * eggbox.width,
    eggbox.y + 0.1 * eggbox.height,
  );
  await page.mouse.down();
  await page.mouse.move(
    eggbox.x + 0.9 * eggbox.width,
    eggbox.y + 0.9 * eggbox.height,
  );
  await page.mouse.up();
  await frame(1);
  assert.equal(await page.evaluate(() => project.detection_region), null);
  await frame(0);
  assert.ok(await page.evaluate(() => project.detection_region));
  await page.locator("#clearRegion").click();
  await page.locator("#experiment").fill("synthetic-egg-run");
  await page.locator("#experiment").press("Tab");
  await page.locator("#eggCount").fill("0");
  await page.locator("#saveCorrection").click();
  await frame(1);
  assert.equal(await page.locator("#eggCount").inputValue(), "");
  await page.locator("#eggCount").fill("12");
  await page.locator("#saveCorrection").click();
  await frame(0);
  assert.equal(await page.locator("#eggCount").inputValue(), "0");
  assert.equal(await page.locator("#wellSelect option").count(), 2);
  await download("training", "egg-training.zip");
  await download("report", "egg-report.zip");
  await download("json", "egg-project.json");
  await page
    .locator("#import")
    .setInputFiles(path.join(output, "ccrt-project.json"));
  await page.waitForFunction(
    () =>
      document
        .querySelector("#status")
        .textContent.includes("same image files") ||
      document.querySelector("#status").textContent.includes("matching study"),
  );
  await page.screenshot({
    path: path.join(output, "eggs.png"),
    fullPage: true,
  });
  execFileSync(
    python,
    [
      "-c",
      `import zipfile,json,hashlib,importlib.util
from pathlib import Path
from PIL import Image
p=Path(${JSON.stringify(output)})
for task,name,expected,eligible in [('ccrt','ccrt',3,2),('eggs','egg',2,2)]:
 with zipfile.ZipFile(p/f'{name}-training.zip') as z:
  assert z.testzip() is None
  m=json.loads(z.read('annotations.json'));assert m['task']==task;assert m['annotated_sample_count']==expected;assert m['eligible_sample_count']==eligible
  for im in m['images']:
   assert hashlib.sha256(z.read(im['path'])).hexdigest()==im['sha256']
   assert z.read(im['path'])==(p/im['original_filename']).read_bytes()
  for a in m['annotations']:
   assert a['label_source']=='manual';assert a['boundary_reviewed'];assert a['split_group']==f'synthetic-{"ccrt" if task=="ccrt" else "egg"}-run'
   import io
   crop=Image.open(io.BytesIO(z.read(a['crop_path'])));assert crop.size==tuple(a['bbox_pixels'][2:]);assert crop.mode=='RGBA'
   source=Image.open(p/a['filename']);x,y,w,h=a['bbox_pixels'];assert crop.getpixel((w//2,h//2))[:3]==source.getpixel((x+w//2,y+h//2))
  if task=='ccrt':
   assert [a['label'] for a in m['annotations']]==['coma','unknown','awake'];assert m['annotations'][1]['boundary_normalized'][0][0]>.2
   second=Image.open(io.BytesIO(z.read(m['annotations'][1]['crop_path'])));assert second.getpixel((0,0))[3]==0
  else:assert [a['label'] for a in m['annotations']]==[0,12]
  dest=p/f'{name}-dataset';z.extractall(dest)
  spec=importlib.util.spec_from_file_location('dataset',dest/'dataset.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rows=list(module.samples(dest));assert len(rows)==eligible;assert all(r['image'].mode=='RGB' for r in rows)
 with zipfile.ZipFile(p/f'{name}-report.zip') as z:
  assert z.testzip() is None;r=json.loads(z.read('report.json'));assert r['kind']==f'{task}-human-report';assert 'well_boundary' not in r['results'][0];assert 'report.html' in z.namelist()
  if task=='ccrt':assert 'egg_count' not in r['results'][0];assert len(r['results'])==1;assert 'maps/frame-000001.png' in z.namelist()
  else:assert 'state' not in r['results'][0];assert [s['Egg count'] for s in r['results']]==[0,12]
with zipfile.ZipFile(p/'ccrt-machine.zip') as z:
 m=json.loads(z.read('machine.json'));assert m['kind']=='ccrt-machine-report';assert m['time_origin']['offset_seconds']==5;assert len(m['results'])==3;assert m['recovery'][0]['ccrt_seconds'] is None;assert m['recovery'][0]['recovery_upper_seconds']==7;assert 'frames.csv' in z.namelist();assert 'project.json' in z.namelist()
print('ZIP integrity, original image hashes, per-frame polygon crops, masks, manual targets, eligibility, Python loader and task-specific reports passed.')
`,
    ],
    { stdio: "inherit" },
  );
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(
    "PASS study browser: separate pages, contextual controls, manual sequence polygon edits, inclusive labels/uncertainty/undo, egg image isolation, reports and training ZIPs. Artifacts:",
    output,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
