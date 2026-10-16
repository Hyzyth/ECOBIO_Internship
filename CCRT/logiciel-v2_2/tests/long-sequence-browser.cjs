/* Exercise the real browser exporter with a 90-minute, partially labelled grid. */
const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  { execFileSync } = require("node:child_process"),
  { step, openOptions } = require("./workflow-helpers.cjs");
(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-long-")),
    python = process.env.PYTHON || "python";
  execFileSync(python, [
    "-c",
    `from PIL import Image
Image.new('RGB',(64,48),(100,110,120)).save(${JSON.stringify(path.join(out, "frame.png"))})`,
  ]);
  const browser = await chromium.launch({
      executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
      headless: true,
      args: ["--no-sandbox"],
    }),
    page = await browser.newPage({ viewport: { width: 1600, height: 1100 } }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  page.setDefaultTimeout(180000);
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page.locator("#files").setInputFiles(path.join(out, "frame.png"));
  await page.waitForFunction(() => image !== null);
  await page.evaluate(() => {
    const original = files[0];
    files = Array.from(
      { length: 5400 },
      (_, i) =>
        new File([original], `frame${String(i + 1).padStart(6, "0")}.png`, {
          type: "image/png",
          lastModified: original.lastModified + i,
        }),
    );
    urls.forEach(URL.revokeObjectURL);
    urls = files.map(URL.createObjectURL);
    project.frames = files.map(frameMetadata);
    project.experiment_id = "long-grid-synthetic";
    project.wells = Array.from({ length: 60 }, (_, i) => ({
      uid: `well-${i}`,
      id: `W${i + 1}`,
      occupancy: i === 59 ? "empty" : "single",
      points: [
        [0.1, 0.1],
        [0.4, 0.1],
        [0.4, 0.4],
        [0.1, 0.4],
      ],
      overrides: {},
      detection: { review_required: false, partial: false },
      review_frames: [],
    }));
    project.wells[0].overrides[100] = [
      [0.2, 0.1],
      [0.5, 0.1],
      [0.5, 0.4],
      [0.2, 0.4],
    ];
    project.records = {};
    const coma = { state: "coma", notes: "" },
      awake = { state: "awake", notes: "" };
    for (let i = 0; i < 5400; i++)
      for (let w = 0; w < 55; w++)
        project.records[C.key(i, `well-${w}`)] = {
          correction: i < 363 ? coma : awake,
        };
    selected = project.wells[0].uid;
    updateDetails();
    draw();
  });
  await step(page, "export");
  const runs = page.locator("#summary tr").first().locator(".spark span");
  assert.equal(await runs.count(), 2);
  assert.match(
    await runs.last().getAttribute("title"),
    /Frames 364–5400: awake/,
  );
  const dimensions = await runs.evaluateAll((spans) =>
    spans.map((s) => s.getBoundingClientRect().width),
  );
  assert.ok(dimensions[1] > 200);
  assert.ok(dimensions[0] < 25);
  await openOptions(page, "trainingOptions");
  await openOptions(page, "comparisonOptions");
  await page.locator("#poorWell").fill("W1");
  await page.locator("#richWell").fill("W2");
  const start = Date.now(),
    download = page.waitForEvent("download");
  await page.locator("#training").click();
  const archive = await download;
  await archive.saveAs(path.join(out, "long.zip"));
  await page.waitForFunction(() => !busy);
  assert.equal(await page.locator("#datasetDownloads a").count(), 1);
  assert.match(
    await page.locator("#status").textContent(),
    /302400 manual samples/,
  );
  assert.match(
    await page.locator("#datasetDownloads a").getAttribute("download"),
    /training\.zip$/,
  );
  console.log(
    `5400-frame/60-well export completed in ${((Date.now() - start) / 1000).toFixed(1)}s; ${fs.statSync(path.join(out, "long.zip")).size} bytes.`,
  );
  // The same File System Access API is exercised on real origin-private disk.
  // A small subset keeps this check focused on destination/error semantics.
  await step(page, "wells");
  await openOptions(page, "resizeOptions");
  await page.locator("#resizeWells").click();
  assert.equal(await page.evaluate(() => project.wells[0].uid), "well-0");
  assert.equal(
    await page.evaluate(() => project.records["0:well-0"].correction.state),
    "coma",
  );
  assert.ok(
    await page.evaluate(() => project.wells[0].overrides[100][0][0] < 0.2),
  );
  assert.ok(
    await page.evaluate(() => C.boundaryNeedsReview(project.wells[0], 0)),
  );
  await page.locator("#confirmSequence").click();
  assert.equal(
    await page.evaluate(() => C.boundaryNeedsReview(project.wells[0], 0)),
    false,
  );
  const folder = await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory(),
      dir = await root.getDirectoryHandle("dataset", { create: true });
    const small = {
      ...project,
      frames: project.frames.slice(0, 3),
      wells: [{ ...project.wells[0], overrides: {} }],
      records: Object.fromEntries(
        Object.entries(project.records).filter(
          ([k]) => Number(k.split(":")[0]) < 3 && k.endsWith(":well-0"),
        ),
      ),
    };
    const result = await TrainingExport.training(
      small,
      files.slice(0, 3),
      () => {},
      () => false,
      { directory: dir },
    );
    const entries = [];
    for await (const [name] of dir.entries()) entries.push(name);
    const saved = JSON.parse(
      await (await (await dir.getFileHandle("project.json")).getFile()).text(),
    );
    const assertBackup =
      !("results" in saved) && Object.keys(saved.records).length === 3;
    const imageDir = await dir.getDirectoryHandle("images");
    let count = 0;
    for await (const _ of imageDir.entries()) count++;
    const partial = await root.getDirectoryHandle("cancelled", {
      create: true,
    });
    let cancelledExport = false;
    try {
      await TrainingExport.training(
        small,
        files.slice(0, 3),
        () => {},
        () => true,
        { directory: partial },
      );
    } catch (e) {
      cancelledExport = /cancelled/.test(e.message);
    }
    let marker = false;
    try {
      await partial.getFileHandle("dataset-complete.json");
      marker = true;
    } catch {}
    return {
      entries,
      count,
      backup: assertBackup,
      cancelledExport,
      marker,
      hasBlob: !!result.blob,
    };
  });
  assert.ok(folder.entries.includes("dataset-complete.json"));
  assert.ok(folder.entries.includes("comparison-plan.json"));
  assert.equal(folder.count, 3);
  assert.ok(folder.backup);
  assert.ok(folder.cancelledExport);
  assert.equal(folder.marker, false);
  assert.equal(folder.hasBlob, false);
  // Force small ZIP parts to check the user-visible fallback links without
  // manufacturing multi-gigabyte research images in CI.
  await page.evaluate(() => {
    files = files.slice(0, 3);
    urls = urls.slice(0, 3);
    project.frames = project.frames.slice(0, 3);
    project.wells = [{ ...project.wells[0], overrides: {} }];
    project.records = Object.fromEntries(
      Object.entries(project.records).filter(
        ([k]) => Number(k.split(":")[0]) < 3 && k.endsWith(":well-0"),
      ),
    );
    const original = TrainingExport.training;
    TrainingExport.training = (p, f, progress, cancel, opts) =>
      original(p, f, progress, cancel, { ...opts, partBytes: 500 });
    updateDetails();
  });
  await step(page, "export");
  await openOptions(page, "trainingOptions");
  await page.locator("#poorWell").fill("");
  await page.locator("#richWell").fill("");
  await page.locator("#training").click();
  await page.waitForFunction(() => !busy);
  assert.ok((await page.locator("#datasetDownloads a").count()) > 1);
  assert.match(
    await page.locator("#datasetDownloads").textContent(),
    /Save every part/,
  );
  const part = page.waitForEvent("download");
  await page.locator("#datasetDownloads a").first().click();
  assert.match((await part).suggestedFilename(), /part-001\.zip$/);
  assert.deepEqual(errors, []);
  await browser.close();
  execFileSync(
    python,
    [
      "-c",
      `from pathlib import Path
import zipfile,json,sys
sys.path.insert(0,str(Path.cwd()))
from tools.model_experiments import Dataset,load_plan
p=Path(${JSON.stringify(out)})
with zipfile.ZipFile(p/'long.zip') as z:
 assert z.testzip() is None
 assert len(z.namelist())==5407
 assert not any(n.startswith('crops/') for n in z.namelist())
 z.extractall(p/'dataset')
d=Dataset(p/'dataset','train');assert len(d.images)==5400;assert len(d.annotations)==302400
assert d.manifest['mode']=='sequence'
assert len(d.geometry)==324000
assert len(load_plan(d,p/'dataset/comparison-plan.json')['runs'])==15
a=d.annotations['f101-well-0'];assert a['boundary_normalized'][0][0]==.2
assert 'f1-well-57' not in d.annotations
assert d.annotations['f5400-well-59']['label']=='empty'
assert d.sample('f1-well-0')[0].shape==(200,200,1)
backup=json.loads((p/'dataset/project.json').read_text());assert 'results' not in backup;assert len(backup['records'])==297000
print('Long export ZIP integrity, shared geometry, incomplete-export guard, immutable 15-run plan and real model input passed.')`,
    ],
    { stdio: "inherit", env: { ...process.env, KERAS_BACKEND: "torch" } },
  );
  console.log(
    "PASS long sequence export and disk-folder/cancellation paths. Artifacts:",
    out,
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
