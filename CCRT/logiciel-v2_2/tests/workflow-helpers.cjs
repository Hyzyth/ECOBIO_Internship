async function step(page, name) {
  await page.locator(`[data-step="${name}"]`).click();
}
async function openOptions(page, ...ids) {
  for (const id of ids) {
    const details = page.locator("#" + id);
    if (!(await details.evaluate((el) => el.open)))
      await details.locator(":scope > summary").click();
  }
}
module.exports = { step, openOptions };
