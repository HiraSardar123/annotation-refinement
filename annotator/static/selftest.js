"use strict";
// Loaded only with ?selftest in the URL: drives the editing tools and reports
// PASS/FAIL lines into <pre id="selftest">. Used to check the app headlessly.
(async () => {
  const out = [];
  const log = (ok, msg) => out.push(`${ok ? "PASS" : "FAIL"} ${msg}`);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let t = 0; t < 200 && !S.res; t++) await wait(100);
  await wait(400);
  const count = (L, id) => { let n = 0; for (let i = 0; i < L.length; i++) if (L[i] === id) n++; return n; };
  const live = () => [...S.res.inst.values()].filter((v) => !v.deleted);
  const rect = (c) => c.getBoundingClientRect();
  const at = (c, x, y) => { const r = rect(c), v = S.view; return { clientX: r.left + v.tx + (x + 0.5) * v.scale, clientY: r.top + v.ty + (y + 0.5) * v.scale, pointerId: 1, button: 0, bubbles: true }; };
  const drag = (c, a, b) => {
    c.dispatchEvent(new PointerEvent("pointerdown", at(c, ...a)));
    c.dispatchEvent(new PointerEvent("pointermove", at(c, ...b)));
    c.dispatchEvent(new PointerEvent("pointerup", at(c, ...b)));
  };
  try {
    log(S.res && S.res.labels.length === S.W * S.H, `image loaded ${S.W}x${S.H}: ${S.human.inst.size} human, ${live().length} in my edit, ${S.extra.inst.size} only-in-JSON`);
    log($("#toolHint").textContent.startsWith("Inspect"), "tool hint shown for the active tool");

    // inspect: click a human object on the left -> card + partner highlight
    setTool("select");
    const hid = [...S.human.inst.keys()][0];
    let hi = -1; for (let i = 0; i < S.human.labels.length; i++) if (S.human.labels[i] === hid) { hi = i; break; }
    cv.left.dispatchEvent(new PointerEvent("pointerdown", at(cv.left, hi % S.W, Math.floor(hi / S.W))));
    cv.left.dispatchEvent(new PointerEvent("pointerup", at(cv.left, hi % S.W, Math.floor(hi / S.W))));
    await wait(60);
    log(S.sel.side === "human" && S.sel.id === hid && $("#selCard").textContent.includes(S.human.inst.get(hid).name), `inspect selected human "${S.human.inst.get(hid).name}" and showed its card`);

    // edit tools do nothing on the left
    setTool("brush"); S.cls = "droplet"; S.brush = 4; S.sel = { side: null, id: 0 };
    const leftBefore = S.res.labels.slice();
    drag(cv.left, [100, 100], [130, 100]);
    log(leftBefore.every((v, i) => v === S.res.labels[i]), "brush on the left image changes nothing");

    // brush on the right
    const before = live().length;
    drag(cv.right, [100, 100], [130, 100]);
    const painted = S.res.labels[100 * S.W + 115], pi = S.res.inst.get(painted);
    log(pi && pi.cls === "droplet" && live().length === before + 1, `brush created "${pi && pi.name}" (${count(S.res.labels, painted)} px)`);
    undo(); log(S.res.labels[100 * S.W + 115] !== painted, "undo removed the stroke");
    redo(); log(S.res.labels[100 * S.W + 115] === painted, "redo restored it");

    // fill gap in box
    setTool("box"); S.cls = "spatter";
    const emptyBefore = count(S.res.labels, 0);
    drag(cv.right, [10, 10], [19, 19]);
    log(count(S.res.labels, 0) <= emptyBefore - 1, `fill gap labelled ${emptyBefore - count(S.res.labels, 0)} empty px`);

    // change class in box (from a class)
    const src = live().find((v) => v.cls === "external_arc") || live()[0];
    let si = -1; for (let i = 0; i < S.res.labels.length; i++) if (S.res.labels[i] === src.id) { si = i; break; }
    const sx = si % S.W, sy = Math.floor(si / S.W);
    setTool("reclass"); S.from = src.cls; S.cls = "weld_pool";
    drag(cv.right, [sx, sy], [Math.min(S.W - 1, sx + 3), Math.min(S.H - 1, sy + 3)]);
    const now = S.res.inst.get(S.res.labels[si]);
    log(now && now.cls === "weld_pool", `change class in box: pixels of "${src.name}" became weld pool`);
    undo();

    // object card actions
    const target = live().find((x) => x.humanId) || live()[0];
    const px = count(S.res.labels, target.id);
    setClass(target.id, "internal_arc");
    log(S.res.inst.get(target.id).cls === "internal_arc" && count(S.res.labels, target.id) === px, `class change kept all ${px} px of "${target.name}"`);
    undo();
    act("delete", target.id);
    log(count(S.res.labels, target.id) === 0, `delete removed "${target.name}"`);
    undo(); log(count(S.res.labels, target.id) === px, "undo brought it back");
    if (target.humanId) {
      act("useHuman", target.id);
      const n = [...S.res.inst.values()].find((x) => !x.deleted && x.humanId === target.humanId);
      log(n && count(S.res.labels, n.id) === count(S.human.labels, target.humanId), `replace with human version: "${target.name}" equals the human object`);
    }

    // copy a missing human object / a JSON-only object
    const missing = [...S.human.inst.values()].find((h) => ![...S.res.inst.values()].some((x) => !x.deleted && x.humanId === h.id));
    if (missing) { copyHuman(missing.id); log(S.sel.side === "result" && count(S.res.labels, S.sel.id) > 0, `copied missing human "${missing.name}" into my edit`); }
    const ex = [...S.extra.inst.values()][0];
    if (ex) { copyExtra(ex.id); log(count(S.res.labels, S.sel.id) > 0, `copied JSON-only "${ex.name}" into my edit (${count(S.res.labels, S.sel.id)} px)`); }

    // change class in box: to background, and from background
    const tgt2 = live().sort((a, b) => count(S.res.labels, b.id) - count(S.res.labels, a.id))[0];
    let ti = -1; for (let i = 0; i < S.res.labels.length; i++) if (S.res.labels[i] === tgt2.id) { ti = i; break; }
    const tx = ti % S.W, ty = Math.floor(ti / S.W);
    setTool("reclass"); S.from = tgt2.cls; S.cls = "background";
    log([...document.querySelectorAll("#classPick button")].some((b) => b.dataset.c === "background"), "change class offers background as a target");
    drag(cv.right, [tx, ty], [Math.min(S.W - 1, tx + 3), Math.min(S.H - 1, ty + 3)]);
    log(S.res.labels[ti] === 0, `change class in box: ${tgt2.name} pixels → background`);
    S.from = "background"; S.cls = tgt2.cls;
    drag(cv.right, [tx, ty], [Math.min(S.W - 1, tx + 3), Math.min(S.H - 1, ty + 3)]);
    log(S.res.labels[ti] !== 0 && S.res.inst.get(S.res.labels[ti]).cls === tgt2.cls, `change class in box: background → ${cname(tgt2.cls)}`);
    setTool("brush");
    log(![...document.querySelectorAll("#classPick button")].some((b) => b.dataset.c === "background"), "brush does not offer background (the eraser does that)");

    // photo view shows the plain image
    setMode("photo");
    const px0 = img.right.data, o = (ty * S.W + tx) * 4;
    log(px0[o] === S.photoRGB[o] && px0[o + 1] === S.photoRGB[o + 1], "photo view shows the camera image unchanged");

    // image adjustments change the displayed photo, reset restores it exactly
    const o2 = (200 * S.W + 480) * 4, orig = [...S.photoOrig.slice(o2, o2 + 3)];
    $("#adjContrast").value = 60; $("#adjContrast").dispatchEvent(new Event("input"));
    $("#adjLocal").value = 70; $("#adjLocal").dispatchEvent(new Event("input"));
    $("#adjBright").value = 30; $("#adjBright").dispatchEvent(new Event("change"));
    await wait(300);
    const adjd = [...img.right.data.slice(o2, o2 + 3)];
    log(adjd.join() !== orig.join(), `brightness/contrast/local contrast change the photo (${orig.join(",")} → ${adjd.join(",")})`);
    $("#adjReset").click(); await wait(300);
    const after = [...img.right.data.slice(o2, o2 + 3)];
    log(after.join() === orig.join() && $("#adjBright").value === "0",
      `Reset image to default restores the original photo (now ${after.join(",")}, slider ${$("#adjBright").value}, adj ${JSON.stringify(S.adj)}, mode ${S.mode}, same buffer ${S.photoRGB === S.photoOrig})`);

    // brush colour = this folder's colour scheme
    setMode("mask");
    const pc = img.right.data, rc = [pc[o], pc[o + 1], pc[o + 2]], want = ccol(tgt2.cls);
    log(rc.join() === want.join(), `mask view paints ${cname(tgt2.cls)} in the folder colour (${want.join(", ")})`);
    log(/Saving to/.test($("#saveTo").textContent), "status bar shows " + $("#saveTo").textContent);

    // brush keeps the exact class colour (selection is only a white outline)
    setMode("mask"); setTool("brush"); S.cls = "droplet"; S.brush = 3; S.sel = { side: null, id: 0 };
    drag(cv.right, [300, 60], [320, 60]);
    const bid = S.res.labels[60 * S.W + 310], bo = (60 * S.W + 310) * 4, bd = img.right.data;
    log(S.sel.id === bid && [bd[bo], bd[bo + 1], bd[bo + 2]].join() === ccol("droplet").join(),
      `painted droplet shows the exact colour (${[bd[bo], bd[bo + 1], bd[bo + 2]].join(", ")}) while selected`);

    // painting a connection merges two droplets into one
    S.sel = { side: null, id: 0 };
    drag(cv.right, [300, 90], [320, 90]);
    const second = S.res.labels[90 * S.W + 310];
    log(second && second !== bid, "second droplet is a separate object");
    S.sel = { side: null, id: 0 };
    drag(cv.right, [310, 60], [310, 90]);
    const a1 = S.res.labels[60 * S.W + 310], a2 = S.res.labels[90 * S.W + 310];
    log(a1 === a2 && S.res.inst.get(a1).cls === "droplet", `painting between them merged both into "${S.res.inst.get(a1).name}"`);

    // explicit merge from the object card
    S.sel = { side: null, id: 0 };
    drag(cv.right, [360, 60], [380, 60]);
    const third = S.res.labels[60 * S.W + 370];
    S.sel = { side: "result", id: a1 }; buildCard();
    $("#mergeWith").value = String(third);
    $("#selCard [data-merge]").click();
    log(S.res.labels[60 * S.W + 370] === S.res.labels[60 * S.W + 310], "Merge with… joined a separate droplet into the selected one");

    // Shift+click two objects, then the Merge button on the card
    setTool("brush"); S.cls = "spatter"; S.brush = 3; S.sel = { side: null, id: 0 }; S.picked = new Set();
    drag(cv.right, [420, 60], [430, 60]); S.sel = { side: null, id: 0 }; S.picked = new Set();
    drag(cv.right, [460, 60], [470, 60]);
    const sA = S.res.labels[60 * S.W + 425], sB = S.res.labels[60 * S.W + 465];
    setTool("select");
    const click = (x, y, shift) => {
      const e = { ...at(cv.right, x, y), shiftKey: shift };
      cv.right.dispatchEvent(new PointerEvent("pointerdown", e)); cv.right.dispatchEvent(new PointerEvent("pointerup", e));
    };
    click(425, 60, false); click(465, 60, true);
    await wait(60);
    log(S.picked.size === 2 && /2 objects selected/.test($("#selCard").textContent) && !!$("#selCard [data-multi=merge]"),
      "Shift+click selected 2 objects and the card shows a Merge button");
    $("#selCard [data-multi=merge]").click();
    log(S.res.labels[60 * S.W + 425] === S.res.labels[60 * S.W + 465] && sA !== sB, "Merge button joined them into one object");

    // mixed classes: merged object takes the chosen class
    const pick2 = live().filter((v) => v.cls !== "spatter")[0];
    let pi2 = -1; for (let i = 0; i < S.res.labels.length; i++) if (S.res.labels[i] === pick2.id) { pi2 = i; break; }
    click(425, 60, false); click(pi2 % S.W, Math.floor(pi2 / S.W), true); await wait(60);
    $("#mergeClass").value = pick2.cls; $("#selCard [data-multi=merge]").click();
    const merged = S.res.inst.get(S.res.labels[60 * S.W + 425]);
    log(merged && merged.cls === pick2.cls && S.res.labels[pi2] === merged.id, `mixed-class merge became one ${cname(pick2.cls)} object`);
    undo(); undo();

    // views and save
    setMode("mask"); setMode("outline"); setMode("overlay");
    $("#compare").checked = true; $("#compare").dispatchEvent(new Event("change"));
    await save(); await wait(400);
    log(!S.dirty && /Saved/.test($("#status").textContent), "saved: " + $("#status").textContent.slice(0, 70));
    log(/Reviewed/.test($("#saveState").textContent), "badge after saving: " + $("#saveState").textContent);

    // Save & next: marks this image and opens the next one
    const k1 = S.key;
    $("#saveNextBtn").click();
    for (let t = 0; t < 60 && S.key === k1; t++) await wait(100);
    await wait(300);
    const p1 = locate(k1);
    log(S.key !== k1 && S.tree[p1.e].images[p1.i].saved, "Save & next saved the image and opened the next one");
    log(/Not reviewed/.test($("#saveState").textContent), "badge on a new image: " + $("#saveState").textContent);

    // Next unreviewed: from here, skip ahead past a saved image
    const k2 = S.key;
    $("#nextOpenBtn").click();
    for (let t = 0; t < 60 && S.key === k2; t++) await wait(100);
    await wait(300);
    const p2 = locate(S.key);
    log(S.key !== k2 && !S.tree[p2.e].images[p2.i].saved, "Next unreviewed opened an image that is not reviewed");
    log(/^○/.test($("#imgSelect").options[$("#imgSelect").selectedIndex].text), "unreviewed images are marked ○ in the image list");

    // leaving the folder with unreviewed images is blocked by the gate
    const e0 = pos().e, keyBefore = S.key;
    $("#expSelect").value = e0 + 1; $("#expSelect").dispatchEvent(new Event("change"));
    await wait(200);
    log(!$("#gate").hidden && S.key === keyBefore && pos().e === e0, `switching folder was blocked: “${$("#gateTitle").textContent}”`);
    const listed = $("#gateList").querySelectorAll("button").length;
    const notDone = S.tree[e0].images.filter((im) => !im.saved).length;
    log(listed === notDone, `the dialog lists all ${notDone} unreviewed images`);
    // arrow keys do nothing while the dialog is open
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    await wait(200);
    log(S.key === keyBefore, "arrow keys are blocked while the dialog is open");

    // Save all as they are -> folder complete -> then the move happens
    $("#gateSaveAll").click();
    for (let t = 0; t < 150 && pos().e === e0; t++) await wait(200);
    await wait(500);
    log(S.tree[e0].images.every((im) => im.saved) && S.tree[e0].published, `Save all marked all ${S.tree[e0].images.length} images reviewed and the folder is complete`);
    log(pos().e === e0 + 1, "after saving all, the tool moved on to the next folder");

    // a folder you have not started does not hold you
    const e1 = pos().e;
    $("#expSelect").value = e1 + 1; $("#expSelect").dispatchEvent(new Event("change"));
    for (let t = 0; t < 60 && pos().e === e1; t++) await wait(100);
    await wait(300);
    log($("#gate").hidden && pos().e === e1 + 1, "leaving a folder you have not started is allowed");

    // start this folder, then "Only what's left"
    const e2 = pos().e;
    await save(); await wait(200);
    $("#onlyLeft").checked = true; $("#onlyLeft").dispatchEvent(new Event("change"));
    for (let t = 0; t < 60 && S.tree[e2].images[pos().i].saved; t++) await wait(100);
    await wait(300);
    const folderOpts = [...$("#expSelect").options].map((o) => +o.value);
    log(folderOpts.length === 1 && folderOpts[0] === e2, `folder list shows only the started folder (${folderOpts.length} shown)`);
    const imgOpts = [...$("#imgSelect").options].map((o) => o.text);
    log(imgOpts.length === S.tree[e2].images.length - 1 && imgOpts.every((t) => t.startsWith("○")),
      `image list shows only the ${imgOpts.length} unreviewed images`);
    const iBefore = pos().i;
    step(1); await wait(800);
    log(pos().e === e2 && pos().i !== iBefore && !S.tree[e2].images[pos().i].saved, "▶ steps to the next unreviewed image");
    $("#onlyLeft").checked = false; $("#onlyLeft").dispatchEvent(new Event("change"));
    log($("#expSelect").options.length === S.tree.length, "switching it off shows every folder again");
  } catch (err) {
    log(false, "exception: " + err.stack);
  }
  const pre = document.createElement("pre"); pre.id = "selftest"; pre.textContent = out.join("\n");
  document.body.appendChild(pre);
})();
