"use strict";
const $ = (id) => document.getElementById(id);
let poseProject = null,
  poseFiles = new Map(),
  poseIndex = 0,
  poseNode = 0,
  poseImage = null,
  poseLoad = 0;
const status = (message) => ($("poseStatus").textContent = message);
function validate(m) {
  if (
    m.kind !== "flyscope-pose-labels" ||
    m.schema_version !== 1 ||
    !Array.isArray(m.nodes) ||
    !m.nodes.length ||
    !m.nodes.every((n) => typeof n === "string" && n.length) ||
    new Set(m.examples?.map((e) => e.id)).size !== m.examples?.length ||
    new Set(m.nodes).size !== m.nodes.length ||
    !Array.isArray(m.examples) ||
    !m.examples.length
  )
    throw Error("Not an anatomical annotation project.");
  for (const e of m.examples) {
    if (e.points.length !== m.nodes.length || !poseFiles.has(e.image_path))
      throw Error("Images or skeleton do not match this project.");
    for (const p of e.points)
      if (
        !["unlabeled", "visible", "occluded"].includes(p.status) ||
        (p.status === "visible" &&
          ![p.x, p.y].every((v) => Number.isFinite(v) && v >= 0 && v <= 1))
      )
        throw Error("Invalid anatomical coordinates.");
  }
  return m;
}
function enableProject(value) {
  for (const id of [
    "poseSave",
    "poseZip",
    "poseReviewed",
    "poseOccluded",
    "poseClear",
    "poseExample",
    "posePrev",
    "poseNext",
  ])
    $(id).disabled = !value;
}
function current() {
  return poseProject?.examples[poseIndex];
}
function draw() {
  const c = $("poseCanvas"),
    ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!poseImage) return;
  ctx.drawImage(poseImage, 0, 0, c.width, c.height);
  const scale = c.width / (c.getBoundingClientRect().width || c.width);
  const edges = poseProject.edges || [
    ["head", "thorax"],
    ["thorax", "abdomen"],
    ...[
      "wingL",
      "wingR",
      "forelegL4",
      "forelegR4",
      "midlegL4",
      "midlegR4",
      "hindlegL4",
      "hindlegR4",
    ].map((n) => ["thorax", n]),
  ];
  if ($("poseLines").checked) {
    ctx.strokeStyle = "#39c8ff";
    ctx.lineWidth = scale;
    for (const [a, b] of edges) {
      const p = current().points[poseProject.nodes.indexOf(a)],
        q = current().points[poseProject.nodes.indexOf(b)];
      if (p?.status !== "visible" || q?.status !== "visible") continue;
      ctx.beginPath();
      ctx.moveTo(p.x * (c.width - 1), p.y * (c.height - 1));
      ctx.lineTo(q.x * (c.width - 1), q.y * (c.height - 1));
      ctx.stroke();
    }
  }
  current().points.forEach((p, i) => {
    if (p.status !== "visible") return;
    ctx.beginPath();
    ctx.fillStyle = i === poseNode ? "#ffd34d" : "#00ec57";
    ctx.arc(
      p.x * (c.width - 1),
      p.y * (c.height - 1),
      Number($("poseDotSize").value) * scale,
      0,
      Math.PI * 2,
    );
    ctx.fill();
    ctx.fillStyle = "#07121c";
    if ($("poseNumbers").checked)
      ctx.fillText(
        String(i + 1),
        p.x * (c.width - 1) + 7,
        p.y * (c.height - 1) - 5,
      );
  });
}
for (const id of ["poseDotSize", "poseLines", "poseNumbers"])
  $(id).oninput = draw;
window.addEventListener("resize", draw);
function controls() {
  const e = current();
  $("poseNodes").replaceChildren(
    ...poseProject.nodes.map((name, i) => {
      const b = document.createElement("button");
      b.textContent = `${i + 1} · ${name} · ${e.points[i].status}`;
      b.className = "secondary";
      b.setAttribute("aria-current", String(i === poseNode));
      b.onclick = () => {
        poseNode = i;
        controls();
        draw();
      };
      return b;
    }),
  );
  $("poseReviewed").checked = e.reviewed;
  $("poseCaption").textContent =
    `${e.well_id} · frame ${e.frame + 1} · ${e.state_context} · ${poseProject.examples.filter((x) => x.reviewed).length}/${poseProject.examples.length} reviewed`;
  $("poseExample").value = String(poseIndex);
  $("posePrev").disabled = poseIndex === 0;
  $("poseNext").disabled = poseIndex === poseProject.examples.length - 1;
}
async function show() {
  const token = ++poseLoad;
  poseImage?.close();
  poseImage = null;
  draw();
  const e = current(),
    file = poseFiles.get(e.image_path);
  const bytes = await file.arrayBuffer();
  const sha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (token !== poseLoad) return;
  if (sha !== e.image_sha256)
    throw Error("Source image changed since project preparation.");
  const bitmap = await createImageBitmap(file);
  if (token !== poseLoad) {
    bitmap.close();
    return;
  }
  poseImage = bitmap;
  $("poseCanvas").width = poseImage.width;
  $("poseCanvas").height = poseImage.height;
  controls();
  draw();
}
function save(name, blob) {
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$("poseFolder").onchange = async (event) => {
  enableProject(false);
  $("poseNodes").replaceChildren();
  try {
    ++poseLoad;
    poseImage?.close();
    poseImage = null;
    poseProject = null;
    draw();
    poseFiles = new Map(
      [...event.target.files].map((f) => [
        f.webkitRelativePath.split("/").slice(1).join("/"),
        f,
      ]),
    );
    const file = poseFiles.get("pose-labels.json");
    if (!file)
      throw Error("Select the folder containing pose-labels.json and images/.");
    poseProject = validate(JSON.parse(await file.text()));
    poseIndex = poseNode = 0;
    $("poseExample").replaceChildren(
      ...poseProject.examples.map(
        (e, i) =>
          new Option(
            `${e.well_id} · ${e.frame + 1} · ${e.state_context}`,
            String(i),
          ),
      ),
    );
    await show();
    enableProject(true);
    controls();
    status("Project loaded. Mark keypoints; save before closing this page.");
  } catch (e) {
    status(e.message);
  }
};
$("poseImport").onchange = async (event) => {
  try {
    if (!poseProject) throw Error("Select the prepared image folder first.");
    const incoming = validate(JSON.parse(await event.target.files[0].text()));
    if (
      JSON.stringify(incoming.nodes) !== JSON.stringify(poseProject.nodes) ||
      incoming.examples.length !== poseProject.examples.length ||
      incoming.examples.some(
        (e, i) =>
          e.id !== poseProject.examples[i].id ||
          e.image_sha256 !== poseProject.examples[i].image_sha256,
      )
    )
      throw Error("Saved annotations belong to different images.");
    // Resume human edits only; immutable source identities come from preparation.
    incoming.examples.forEach((e, i) => {
      poseProject.examples[i].points = e.points;
      poseProject.examples[i].reviewed = e.reviewed === true;
    });
    await show();
    status("Saved annotations restored.");
  } catch (e) {
    status(e.message);
  }
};
$("poseCanvas").onclick = (event) => {
  if (!poseImage) return;
  const rect = event.target.getBoundingClientRect();
  current().points[poseNode] = {
    status: "visible",
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
  current().reviewed = false;
  poseNode = Math.min(poseNode + 1, poseProject.nodes.length - 1);
  controls();
  draw();
};
$("poseOccluded").onclick = () => {
  current().points[poseNode] = { status: "occluded", x: null, y: null };
  current().reviewed = false;
  poseNode = Math.min(poseNode + 1, poseProject.nodes.length - 1);
  controls();
  draw();
};
$("poseClear").onclick = () => {
  current().points[poseNode] = { status: "unlabeled", x: null, y: null };
  current().reviewed = false;
  controls();
  draw();
};
$("poseReviewed").onchange = () => {
  if (
    $("poseReviewed").checked &&
    current().points.some((p) => p.status === "unlabeled")
  ) {
    status("Mark or explicitly occlude every node before confirming review.");
    $("poseReviewed").checked = false;
    return;
  }
  current().reviewed = $("poseReviewed").checked;
  controls();
};
for (const [id, delta] of [
  ["posePrev", -1],
  ["poseNext", 1],
])
  $(id).onclick = async () => {
    try {
      poseIndex += delta;
      await show();
    } catch (e) {
      status(e.message);
    }
  };
$("poseExample").onchange = async (e) => {
  try {
    poseIndex = Number(e.target.value);
    await show();
  } catch (e) {
    status(e.message);
  }
};
$("poseSave").onclick = () =>
  save(
    "pose-labels.json",
    new Blob([JSON.stringify(poseProject, null, 2)], {
      type: "application/json",
    }),
  );
$("poseZip").onclick = async () => {
  try {
    $("poseZip").disabled = true;
    const entries = [
      { name: "pose-labels.json", text: JSON.stringify(poseProject, null, 2) },
      ...poseProject.examples.map((e) => ({
        name: e.image_path,
        blob: poseFiles.get(e.image_path),
      })),
    ];
    save("pose-labels.zip", await TrainingExport.zip(entries));
    status(
      "Labeled dataset exported. Extract it before running the pose trainer.",
    );
  } catch (e) {
    status(e.message);
  } finally {
    $("poseZip").disabled = false;
  }
};
window.addEventListener("beforeunload", (e) => {
  if (poseProject) {
    e.preventDefault();
    e.returnValue = "";
  }
});
