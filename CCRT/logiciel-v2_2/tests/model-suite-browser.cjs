const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");
const { step, openOptions } = require("./workflow-helpers.cjs");

(async () => {
  const source = process.env.SUITE_SMOKE_DIR;
  assert.ok(
    source,
    "Set SUITE_SMOKE_DIR to an actual validate_model_suite.py output.",
  );
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "flyscope-suite-browser-"));
  const files = [1, 2, 3, 4, 5, 6].map((f) =>
    path.join(
      source,
      "fixtures/test/images",
      `frame-${String(f).padStart(6, "0")}.png`,
    ),
  );
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
  await page.goto((process.env.APP_URL || "http://127.0.0.1:5000") + "/ccrt");
  await page.waitForFunction(() =>
    models.some((m) => m.architecture === "sleap-rf" && m.available),
  );
  await page.locator("#files").setInputFiles(files);
  await page.waitForFunction(() => image !== null);
  await page.locator("#interval").fill("2");
  await page.locator("#interval").press("Tab");
  await page.locator("#analysisStart").fill("3");
  await page.locator("#analysisStart").press("Tab");
  await page.waitForFunction(() => index === 2 && image !== null);
  await openOptions(page, "timingOptions");
  await page.locator("#recoveryDelay").fill("10");
  await page.locator("#recoveryDelay").press("Tab");
  await page
    .locator("#gridSpecies")
    .evaluate((el) => (el.closest("details").open = true));
  await page.locator("#gridSpecies").fill("D. melanogaster");
  await page.locator("#gridSpecies").press("Tab");
  await page.locator("#inputContinue").click();
  await page.evaluate(() => {
    addWell([
      [35 / 600, 35 / 400],
      [165 / 600, 35 / 400],
      [165 / 600, 165 / 400],
      [35 / 600, 165 / 400],
    ]);
    addWell([
      [435 / 600, 35 / 400],
      [565 / 600, 35 / 400],
      [565 / 600, 165 / 400],
      [435 / 600, 165 / 400],
    ]);
  });
  await page.locator("#confirmSequence").click();
  await page.locator("#wellsContinue").click();
  assert.equal(await page.locator("#timeline").getAttribute("min"), "2");
  await page.locator("#wellSelect").selectOption({ index: 1 });
  await page.locator("#occupancy").selectOption("single");
  await page.locator("#comaUntil").click();
  await page.locator("#next").click();
  await page.waitForFunction(() => index === 3 && image !== null);
  await page.locator("#awakeFrom").click();
  assert.equal(await page.locator("#nextLabel").isVisible(), false);
  await page.locator("#wellSelect").selectOption({ index: 2 });
  await page.locator("#occupancy").selectOption("empty");
  assert.match(await page.locator("#prediction").textContent(), /EMPTY/);
  await openOptions(page, "modelOptions");
  for (const kind of ["resnet18", "sleap-rf", "deepethogram"]) {
    const model = await page.evaluate(
      (kind) =>
        models.find(
          (m) =>
            m.architecture === kind && m.name.endsWith("all-annotated-wells"),
        ),
      kind,
    );
    assert.ok(model?.available);
    await page.locator("#model").selectOption(model.id);
    assert.equal(await page.locator("#useModelPolicy").isChecked(), true);
    assert.equal(await page.locator("#manualPolicyOptions").isVisible(), false);
    assert.match(
      await page.locator("#policyInfo").textContent(),
      /could not be validated/,
    );
    if (kind === "resnet18") {
      await page.locator("#useModelPolicy").uncheck();
      await openOptions(page, "manualPolicyOptions");
      await page.locator("#awakeFrames").fill("1");
      await page.locator("#awakeFrames").press("Tab");
      await page.locator("#inferMatchingGaps").check();
      const manual = await page.evaluate(() => project.recovery_policy);
      assert.equal(manual.source, "manual_override");
      assert.equal(manual.awake_consecutive_frames, 1);
      assert.equal(manual.automatic_confirmation, true);
      assert.equal(manual.awake_confirmation_seconds, undefined);
      assert.equal(manual.infer_matching_gaps, true);
      await page.locator("#useModelPolicy").check();
      assert.equal(
        await page.locator("#manualPolicyOptions").isVisible(),
        false,
      );
      assert.equal(
        await page.evaluate(
          () => project.recovery_policy.automatic_confirmation,
        ),
        false,
      );
    }
    await page.locator("#analyze").click();
    await page.waitForFunction(
      (version) =>
        !busy &&
        Object.values(project.records).filter(
          (r) => r.model?.version === version,
        ).length === 8,
      model.version,
      { timeout: 120000 },
    );
    const data = await page.evaluate(() => C.exportProject(project));
    assert.ok(
      Object.keys(data.records).every((k) => Number(k.split(":")[0]) >= 2),
    );
    assert.ok(
      Object.values(data.records).every(
        (r) => Number.isFinite(r.raw_confidence) && r.raw_prediction,
      ),
    );
    assert.equal(
      data.records[`2:${data.wells[0].uid}`].correction.state,
      "coma",
    );
    assert.equal(
      data.records[`3:${data.wells[0].uid}`].correction.state,
      "awake",
    );
  }
  await step(page, "export");
  const recovery = await page.evaluate(() =>
    C.recoveryMetrics(project, project.wells[0]),
  );
  assert.equal(recovery.first_awake_frame, 4);
  assert.equal(recovery.ccrt_seconds, 16);
  async function download(id, name) {
    const pending = page.waitForEvent("download");
    await page.locator("#" + id).click();
    await (await pending).saveAs(path.join(out, name));
    await page.waitForFunction(() => !busy);
  }
  await download("json", "project.json");
  await download("report", "report.zip");
  await openOptions(page, "trainingOptions");
  await download("training", "train.zip");
  await page.locator("#datasetTest").check();
  await download("training", "test.zip");
  await step(page, "images");
  await page.locator("#inputKind").selectOption("trimmed");
  assert.equal(await page.locator("#timeConfirmed").isChecked(), false);
  assert.equal(
    await page.evaluate(
      () => C.recoveryMetrics(project, project.wells[0]).ccrt_seconds,
    ),
    null,
  );
  await openOptions(page, "restoreOptions");
  await page.locator("#import").setInputFiles(path.join(out, "project.json"));
  await page.waitForFunction(
    () =>
      document.body.dataset.workflow === "review" &&
      project.input_kind === "raw",
  );
  assert.equal(await page.evaluate(() => project.analysis_start_index), 2);
  assert.equal(await page.evaluate(() => project.species), "D. melanogaster");
  await step(page, "wells");
  await page.evaluate(() => {
    $("scope").value = "frame";
    const w = project.wells[0];
    saveGeometry(
      w,
      C.geometry(w, index).map(([x, y]) => [x + 0.001, y]),
    );
  });
  assert.equal(
    await page.evaluate(
      () => project.records[C.key(3, project.wells[0].uid)].correction.state,
    ),
    "awake",
  );
  assert.equal(
    await page.evaluate(() =>
      C.analysisIndices(project)
        .filter((i) => i >= index)
        .every((i) => !project.records[C.key(i, project.wells[0].uid)].model),
    ),
    true,
  );
  await step(page, "images");
  await page.locator("#analysisStart").fill("4");
  await page.locator("#analysisStart").press("Tab");
  await page.waitForFunction(() => index === 3 && image !== null);
  assert.equal(
    await page.evaluate(() =>
      Object.values(project.records).every((r) => !r.model?.requires_sequence),
    ),
    true,
  );
  assert.equal(
    await page.evaluate(
      () => project.records[C.key(3, project.wells[0].uid)].correction.state,
    ),
    "awake",
  );
  await page.locator("#mode").selectOption("single");
  await page.locator("#files").setInputFiles(files[0]);
  await page.waitForFunction(() => image !== null);
  await page.locator("#inputContinue").click();
  await page.locator("#whole").click();
  await page.locator("#wellsContinue").click();
  await openOptions(page, "modelOptions");
  assert.equal(
    await page
      .locator("#model option")
      .evaluateAll((opts) =>
        opts
          .filter((o) => o.textContent.startsWith("deepethogram"))
          .every((o) => o.disabled),
      ),
    true,
  );
  assert.deepEqual(errors, []);
  await browser.close();
  execFileSync(
    process.env.PYTHON || "python",
    [
      "-c",
      `
import json, zipfile
from pathlib import Path
from backend.services.dataset_format import hydrate_manifest
p=Path(${JSON.stringify(out)})
for role in ['train','test']:
 with zipfile.ZipFile(p/f'{role}.zip') as z:
  assert z.testzip() is None
  m=hydrate_manifest(json.loads(z.read('annotations.json')))
  assert m['analysis_start_index']==2 and m['analysis_frame_count']==4
  assert m['mode']=='sequence' and m['frame_interval_seconds']==2
  assert [im['frame_number'] for im in m['images']]==[3,4,5,6]
  assert len(m['annotations'])==8 and all(a['frame_index']>=2 for a in m['annotations'])
  assert all(a['species']=='D. melanogaster' for a in m['annotations'] if a['label']!='empty')
  assert m['time_origin']['offset_seconds']==10
  assert m['role']==role
  if role=='test': assert 'comparison-plan.json' not in z.namelist() and not any(n.startswith('crops/') for n in z.namelist())
with zipfile.ZipFile(p/'report.zip') as z:
 csv=z.read('results.csv').decode()
 assert 'First awake frame' in csv and 'Frame ID' in csv and 'Time in coma' not in csv
print('PASS actual suite app inference, raw/trimmed timing, sustained recovery, preserved annotations, setup-free training/test exports and project restore:',p)
`,
    ],
    { stdio: "inherit" },
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
