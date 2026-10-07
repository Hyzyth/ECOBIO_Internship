const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process");
(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-experiments-")),
    python = process.env.PYTHON || "python";
  execFileSync(python, [
    "-c",
    `from pathlib import Path
import cv2,numpy as np
p=Path(${JSON.stringify(out)})
for group in ['train','test']:
 (p/group).mkdir()
 for f in range(3):
  im=np.full((500,800,3),150 if group=='train' else 165,np.uint8)
  for row in range(2):
   for col in range(3):
    x,y=150+250*col,130+240*row
    cv2.circle(im,(x,y),70,(200,200,200),3)
    if row or col!=2:cv2.ellipse(im,(x+f*2,y),(30,12),30+f*3,0,360,(20,50,100),-1)
  cv2.imwrite(str(p/group/f'frame{f}.png'),im)
`,
  ]);
  const browser = await chromium.launch({
      executablePath: "/usr/bin/chromium",
      headless: true,
      args: ["--no-sandbox"],
    }),
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("dialog", (d) => d.accept());
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page.waitForFunction(
    () => document.querySelector("#model").options.length >= 3,
  );
  async function download(id, name) {
    const promise = page.waitForEvent("download");
    await page.locator("#" + id).click();
    const d = await promise;
    await d.saveAs(path.join(out, name));
    await page.waitForFunction(() => !busy);
  }
  async function setup(group) {
    await page
      .locator("#files")
      .setInputFiles(
        [0, 1, 2].map((f) => path.join(out, group, `frame${f}.png`)),
      );
    await page.waitForFunction(() => image !== null);
    await page.locator("#experiment").fill("synthetic-" + group);
    await page.locator("#experiment").press("Tab");
    await page.evaluate(() => {
      for (let row = 0; row < 2; row++)
        for (let col = 0; col < 3; col++) {
          const x = 150 + 250 * col,
            y = 130 + 240 * row;
          addWell([
            [(x - 70) / 800, (y - 70) / 500],
            [(x + 70) / 800, (y - 70) / 500],
            [(x + 70) / 800, (y + 70) / 500],
            [(x - 70) / 800, (y + 70) / 500],
          ]);
        }
    });
    await page.locator("#confirmSequence").click();
    for (const [n, label] of [
      [0, "coma"],
      [1, "awake"],
    ]) {
      await page.locator("#wellSelect").selectOption({ index: n + 1 });
      await page.locator("#occupancy").selectOption("single");
      await page.locator("#rangeStart").fill("1");
      await page.locator("#rangeEnd").fill("3");
      await page.locator("#rangeState").selectOption(label);
      await page.locator("#applyRange").click();
    }
    await page.locator("#wellSelect").selectOption({ index: 3 });
    await page.locator("#occupancy").selectOption("empty");
  }
  await setup("train");
  await page.locator("#comparisonSeed").fill("77");
  await page.locator("#poorWell").fill("W1");
  await page.locator("#richWell").fill("W2");
  await page.locator("#model").selectOption("legacy-model1");
  await page.locator("#analyze").click();
  await page.waitForFunction(
    () =>
      !busy &&
      Object.values(project.records).filter((r) => r.prediction).length === 18,
    {},
    { timeout: 90000 },
  );
  assert.equal(
    await page.evaluate(
      () => project.records[C.key(0, project.wells[2].uid)].prediction,
    ),
    "empty",
  );
  assert.equal(
    await page.evaluate(
      () => project.records[C.key(0, project.wells[0].uid)].correction.state,
    ),
    "coma",
  );
  assert.equal(
    await page.evaluate(() => C.trainingAnnotations(project).length),
    9,
  );
  await download("training", "train.zip");
  await page.locator("#layout").selectOption("individual");
  assert.equal(await page.locator("#layout").inputValue(), "grid");
  await setup("test");
  await page.locator("#datasetTest").check();
  assert.equal(await page.locator("#comparisonOptions").isVisible(), false);
  await download("training", "test.zip");
  // Single-well mode keeps a stable identity when replacing the boundary.
  await page.locator("#mode").selectOption("single");
  await page.locator("#layout").selectOption("individual");
  await page
    .locator("#files")
    .setInputFiles(path.join(out, "test", "frame0.png"));
  await page.waitForFunction(() => !busy && image !== null);
  await page.locator("#whole").click();
  assert.equal(await page.evaluate(() => project.wells.length), 1);
  const uid = await page.evaluate(() => project.wells[0].uid);
  await page.locator("#whole").click();
  assert.equal(await page.evaluate(() => project.wells.length), 1);
  assert.equal(await page.evaluate(() => project.wells[0].uid), uid);
  await page.screenshot({
    path: path.join(out, "single-well.png"),
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  await browser.close();
  execFileSync(
    python,
    [
      "-c",
      `from pathlib import Path
import zipfile,json
p=Path(${JSON.stringify(out)})
for role in ['train','test']:
 with zipfile.ZipFile(p/f'{role}.zip') as z:
  assert z.testzip() is None
  m=json.loads(z.read('annotations.json'));assert m['role']==role;assert all(m['coverage'][k]==v for k,v in {'detected_wells':6,'annotated_wells':3,'eligible_wells':3}.items());assert len(m['annotations'])==9;assert len(m['images'])==3
  assert sum(a['label']=='empty' for a in m['annotations'])==3;assert all(a['label_source'] in ['manual','manual_occupancy'] for a in m['annotations'])
  if role=='test':assert not any(name.startswith('crops/') for name in z.namelist());assert 'project.json' not in z.namelist();assert 'comparison-plan.json' not in z.namelist()
  else:
   plan=json.loads(z.read('comparison-plan.json'));assert plan['seed']==77;assert plan['runs'][-2]['id']=='poor-poses';assert plan['runs'][-1]['id']=='rich-poses';assert len(plan['runs'])==33
   assert len(plan['skipped'])==4
  z.extractall(p/role/'dataset')
print('Partial grid train/test ZIPs, manual empty targets and fixed well plans passed.')
`,
    ],
    { stdio: "inherit" },
  );
  if (process.env.RUN_MODEL_TRAINING === "1") {
    execFileSync(
      python,
      [
        path.resolve(__dirname, "../tools/model_experiments.py"),
        "compare",
        "--train",
        path.join(out, "train/dataset"),
        "--test",
        path.join(out, "test/dataset"),
        "--models",
        "legacy-model1",
        "legacy-model2",
        "--runs",
        "all-annotated-wells",
        "poor-poses",
        "rich-poses",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--output",
        path.join(out, "trained-results"),
      ],
      { stdio: "inherit" },
    );
    execFileSync(
      python,
      [
        "-c",
        `import json,os
from pathlib import Path
from backend.services.keras_adapter import registered_adapters
p=Path(${JSON.stringify(out)})/'trained-results'
selections={}
for f in p.glob('legacy-*/*/model.json'):
 m=json.loads(f.read_text());run=m['selection']['id'];ids=m['selection']['annotation_ids']
 if run in selections:assert selections[run]==ids
 selections[run]=ids
 if run=='all-annotated-wells':assert m['occupancy_head']['training_examples']==9
 else:assert m['occupancy_head'] is None
assert len(list(p.glob('legacy-*/*/model.keras')))==6
os.environ['FLYSCOPE_MODELS_DIR']=str(p);adapters=registered_adapters();assert len(adapters)==8
trained=next(a for k,a in adapters.items() if k.startswith('trained-'))
content=(Path(${JSON.stringify(out)})/'test/dataset/images/frame-000001.png').read_bytes()
r=trained.predict(content,[{'id':'w','points':[[0,0],[1,0],[1,1],[0,1]]}]);assert len(r)==1
print('PASS both model trainers, identical full/poor/rich selections, learned occupancy and registered artifact prediction.')
`,
      ],
      { stdio: "inherit", cwd: path.resolve(__dirname, "..") },
    );
  }
  console.log("PASS experiment browser. Artifacts:", out);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
