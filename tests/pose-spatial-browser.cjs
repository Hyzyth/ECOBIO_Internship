/* Real UI round trip plus pixel-exact dataset export. Synthetic clicks test
   software only; they are deliberately not biological anatomical ground truth. */
const { chromium } = require("playwright"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  os = require("node:os"),
  { execFileSync } = require("node:child_process");
(async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-pose-spatial-")),
    python = process.env.PYTHON || "python";
  execFileSync(python, [
    "-c",
    `from tests.suite_fixtures import grids
from flyscope.data.pose_labels import prepare
from pathlib import Path
p=Path(${JSON.stringify(out)})
train,test=grids(p/'fixtures')
prepare(train,p/'pose',count=8,size=64,wells=['W1','W2'])`,
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
  const url = process.env.APP_URL || "http://127.0.0.1:5000";
  await page.goto(url + "/pose");
  await page.locator("#poseFolder").setInputFiles(path.join(out, "pose"));
  await page.waitForFunction(() => poseProject !== null && poseImage !== null);
  for (const well of ["W1", "W2"]) {
    const index = await page.evaluate(
      (w) => poseProject.examples.findIndex((e) => e.well_id === w),
      well,
    );
    await page.locator("#poseExample").selectOption(String(index));
    await page.waitForFunction(
      (i) =>
        poseIndex === i &&
        document
          .querySelector("#poseCaption")
          .textContent.startsWith(poseProject.examples[i].well_id),
      index,
    );
    await page.locator("#poseNodes button").nth(0).click();
    await page.locator("#poseCanvas").click({ position: { x: 180, y: 180 } });
    await page.locator("#poseCanvas").click({ position: { x: 250, y: 250 } });
    for (let i = 2; i < 13; i++) await page.locator("#poseOccluded").click();
    await page.locator("#poseReviewed").check();
  }
  const saved = await page.evaluate(() => JSON.stringify(poseProject));
  fs.writeFileSync(path.join(out, "saved.json"), saved);
  await page.locator("#poseClear").click();
  await page.locator("#poseImport").setInputFiles(path.join(out, "saved.json"));
  await page.waitForFunction(() =>
    document.querySelector("#poseStatus").textContent.includes("restored"),
  );
  assert.equal(
    await page.evaluate(
      () => poseProject.examples.filter((e) => e.reviewed).length,
    ),
    2,
  );
  const download = page.waitForEvent("download");
  await page.locator("#poseZip").click();
  await (await download).saveAs(path.join(out, "labeled.zip"));
  await page.goto(url + "/ccrt");
  await page
    .locator("#files")
    .setInputFiles(
      [0, 1].map((i) =>
        path.join(
          out,
          "fixtures/train/images",
          `frame-${String(i + 1).padStart(6, "0")}.png`,
        ),
      ),
    );
  await page.waitForFunction(() => image !== null && !busy);
  assert.equal(await page.locator("#cropDataset").isChecked(), true);
  const result = await page.evaluate(async () => {
    addWell([
      [0.2, 0.2],
      [0.6, 0.2],
      [0.6, 0.6],
      [0.2, 0.6],
    ]);
    const w = project.wells[0];
    w.occupancy = "single";
    w.detection.review_required = false;
    w.overrides[1] = [
      [0.25, 0.25],
      [0.7, 0.25],
      [0.7, 0.65],
      [0.25, 0.65],
    ];
    C.applyStateRange(project, w.uid, 0, 1, "coma");
    const before = JSON.stringify(project);
    const result = await TrainingExport.training(
      project,
      files,
      () => {},
      () => false,
      { cropToWells: true, cropPadding: 16, cropEncoding: "png" },
    );
    const bytes = new Uint8Array(await result.blob.arrayBuffer());
    return {
      base64: btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join("")),
      unchanged: JSON.stringify(project) === before,
    };
  });
  assert.equal(result.unchanged, true);
  fs.writeFileSync(
    path.join(out, "cropped.zip"),
    Buffer.from(result.base64, "base64"),
  );
  assert.deepEqual(errors, []);
  await browser.close();
  execFileSync(
    python,
    [
      "-c",
      `from pathlib import Path
import zipfile,json,cv2,numpy as np
from flyscope.data.datasets import Dataset
from flyscope.data.pose_labels import load_examples
from flyscope.training.pose_quality import split_examples
p=Path(${JSON.stringify(out)})
with zipfile.ZipFile(p/'labeled.zip') as z:z.extractall(p/'labeled')
nodes,examples,_=load_examples(p/'labeled/pose-labels.json',64)
assert len(examples)==2 and all(e[2].sum()==2 for e in examples)
a,b,split=split_examples(examples);assert split['unit']=='whole individual'
with zipfile.ZipFile(p/'cropped.zip') as z:z.extractall(p/'crop')
data=Dataset(p/'crop','train');m=data.manifest
assert m['spatial_export']['encoding']=='png'
backup=json.loads((p/'crop/project.json').read_text());assert backup['wells'][0]['points'][0]==[.2,.2]
for im in m['images']:
 c=im['spatial_crop'];original=cv2.imread(str(p/'fixtures/train'/im['path']))
 x,y,w,h=[round(c[k]) for k in ['source_x','source_y','source_width','source_height']]
 np.testing.assert_array_equal(data.image(im['path']),original[y:y+h,x:x+w])
 assert im['source_sha256'] in data.image_hashes
for frame in m['wells'][0]['frames']:
 expected=backup['wells'][0].get('overrides',{}).get(str(frame['frame_index']),backup['wells'][0]['points'])
 r=m['spatial_export']['normalized_region']
 np.testing.assert_allclose(np.array(frame['points'])*r[2:]+r[:2],expected)
print('PASS manual anatomy UI/export/resume, independent-fly split, pixel-exact spatial export, reversible geometry and source hashes')`,
    ],
    { stdio: "inherit" },
  );
  if (process.env.RUN_MODEL_TRAINING === "1")
    execFileSync(
      python,
      [
        "tools/sleap_pose.py",
        "train",
        "--labels",
        path.join(out, "labeled/pose-labels.json"),
        "--parent",
        process.env.POSE_SMOKE_MODEL,
        "--output",
        path.join(out, "pose-finetuned"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--device",
        "cpu",
      ],
      { stdio: "inherit" },
    );
  console.log("Artifacts:", out);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
