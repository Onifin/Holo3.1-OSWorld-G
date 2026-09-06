"""Review the application label of every OSWorld-G task, one task per screen.

The unit is the task, not the screenshot: a screenshot holding two applications
-- GIMP full-screen with the Settings window over it, a Writer document behind a
Chrome window -- has no single answer, and only the task's own target says which
application it belongs to. So each screen shows one task: its instruction, its
target box drawn on the screenshot, and the label it currently carries, which you
either approve or replace.

    uv run --with opencv-python --with numpy validate_labels.py

    uv run --with opencv-python --with numpy validate_labels.py --only null
    uv run --with opencv-python --with numpy validate_labels.py --only GIMP

Keys
    a       approve the current label      1..8  replace it       0  set null
    A       apply the label you just set to every task on this screenshot
    SPACE / n / ->   skip forward          b / <-  back
    u       clear this task's verdict
    s       save now                       q / ESC  save and quit

Labels are read from and written back to `benchmark/task_labels.json` (one entry
per task id), and verdicts to `benchmark/task_labels_review.json`; both are backed
up to `.bak` on every save. It opens at the first task with no verdict, so the
pass can be done in sittings, and nothing is written until you save.
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

CATEGORIES = [
    "Chrome",
    "LibreOffice Impress",
    "LibreOffice Calc",
    "LibreOffice Writer",
    "GIMP",
    "VsCode",
    "VLC",
    "OS",
]

BAR = (0, 0, 900, 34)
MAX_W, MAX_H = 1460, 690
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW = (120, 255, 120), (80, 80, 255), (80, 220, 255)
GREY, WHITE, BOX = (150, 150, 150), (235, 235, 235), (0, 215, 255)


def wrap(text, width_px, scale, thick=1):
    """Greedy word wrap against the pixel width the frame actually has."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > width_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_target(body, item, scale):
    """Outline the ground-truth target, which is what says who owns the task."""
    if item["box_type"] == "refusal":
        return
    c = item["box_coordinates"]
    if item["box_type"] == "bbox":
        x, y, w, h = (v * scale for v in c)
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(body, p1, p2, BOX, 2)
        cv2.rectangle(body, (p1[0] - 6, p1[1] - 6), (p2[0] + 6, p2[1] + 6), BOX, 1)
    else:
        pts = np.array([[int(c[i] * scale), int(c[i + 1] * scale)]
                        for i in range(0, len(c), 2)], np.int32)
        cv2.polylines(body, [pts], True, BOX, 2)


def build_frame(img, item, meta):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h)
    body = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    draw_target(body, item, scale)
    width = body.shape[1]

    bar = img[BAR[1]:BAR[3], BAR[0]:min(BAR[2], w)]
    if bar.size:
        bs = min(width / bar.shape[1], 2.2)
        bar = cv2.resize(bar, (int(bar.shape[1] * bs), int(bar.shape[0] * bs)),
                         interpolation=cv2.INTER_NEAREST)
        bar = bar[:, :width]

    inst = wrap(item["instruction"], width - 24, 0.56)[:3]
    y = 24
    rows = [("head", 1), ("label", 1), ("inst", len(inst)), ("keys", 1)]
    head_h = 24 + 34 + 8 + 22 * len(inst) + 30
    bar_h = bar.shape[0] + 8 if bar.size else 0
    frame = np.full((head_h + bar_h + body.shape[0], width, 3), 24, np.uint8)
    if bar.size:
        frame[head_h + 4:head_h + 4 + bar.shape[0], :bar.shape[1]] = bar
    frame[head_h + bar_h:, :] = body

    cv2.putText(frame, f"[{meta['i']+1}/{meta['n']}] {item['id']}  {item['image_path']}"
                       f"  ({item['box_type']})   "
                       f"ok:{meta['n_ok']}  alterados:{meta['n_changed']}  "
                       f"faltam:{meta['n_todo']}",
                (12, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    y += 34

    col = YELLOW if meta["changed"] else WHITE
    cv2.putText(frame, meta["label"], (12, y), FONT, 0.9, col, 2, cv2.LINE_AA)
    if meta["changed"]:
        t = f"era: {meta['seed']}"
        cv2.putText(frame, t, (16 + cv2.getTextSize(meta["label"], FONT, 0.9, 2)[0][0], y),
                    FONT, 0.5, GREY, 1, cv2.LINE_AA)
    if meta["verdict"]:
        txt, c = ("APROVADA", GREEN) if meta["verdict"] == "ok" else ("ALTERADA", YELLOW)
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.8, 2)
        cv2.putText(frame, txt, (width - tw - 16, y), FONT, 0.8, c, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width - 1, frame.shape[0] - 1), c, 3)
    y += 8

    for line in inst:
        y += 22
        cv2.putText(frame, line, (12, y), FONT, 0.56, (200, 220, 255), 1, cv2.LINE_AA)

    keys = "  ".join(f"{i+1}:{c.replace('LibreOffice ', 'LO ')}"
                     for i, c in enumerate(CATEGORIES))
    cv2.putText(frame, f"a:aprovar  {keys}  0:null  A:toda a tela  "
                       "espaco:pular  b:voltar  u:limpar  s:salvar  q:sair",
                (12, head_h - 10), FONT, 0.42, GREY, 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="../benchmark/OSWorld-G.json")
    ap.add_argument("--images", default="../benchmark/images")
    ap.add_argument("--task_labels", default="../benchmark/task_labels.json")
    ap.add_argument("--review", default="../benchmark/task_labels_review.json")
    ap.add_argument("--only", default=None,
                    help="Review one category only; 'null' for the unlabelled ones.")
    ap.add_argument("--all", action="store_true",
                    help="Start at the first task rather than the first unreviewed.")
    args = ap.parse_args()

    items = json.load(open(args.annotations))
    if not os.path.exists(args.task_labels):
        raise SystemExit(f"{args.task_labels} não existe -- ele é o ponto de partida "
                         f"da revisão; crie-o com um rótulo por tarefa antes de rodar")
    labels = json.load(open(args.task_labels))
    missing = [x["id"] for x in items if x["id"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} tarefas sem rótulo em {args.task_labels} "
                         f"(ex.: {missing[0]})")
    # Whatever the file said when this sitting opened is what "era:" compares to,
    # so a change made now is visible until it is saved and the file is reopened.
    seed = dict(labels)
    review = json.load(open(args.review)) if os.path.exists(args.review) else {}

    if args.only is not None:
        want = None if args.only.lower() == "null" else args.only
        items = [x for x in items if labels[x["id"]] == want]
        if not items:
            raise SystemExit(f"nenhuma tarefa com a categoria {args.only!r}")

    i = 0
    if not args.all:
        todo = [k for k, x in enumerate(items) if x["id"] not in review]
        i = todo[0] if todo else 0
    dirty = False
    last_set = None

    def save():
        nonlocal dirty
        for path, payload in ((args.task_labels, labels), (args.review, review)):
            if os.path.exists(path):
                shutil.copy(path, path + ".bak")
            with open(path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        n_ch = sum(1 for k in labels if labels[k] != seed[k])
        print(f"salvo: {args.task_labels} e {args.review}  "
              f"({sum(v == 'ok' for v in review.values())} aprovadas, {n_ch} alteradas)")
        dirty = False

    cache = {}
    cv2.namedWindow("labels", cv2.WINDOW_NORMAL)
    while True:
        item = items[i]
        if item["image_path"] not in cache:
            cache.clear()
            cache[item["image_path"]] = cv2.imread(
                os.path.join(args.images, item["image_path"]))
        img = cache[item["image_path"]]
        if img is None:
            print(f"não consegui abrir {item['image_path']}, pulando")
            i = min(i + 1, len(items) - 1)
            continue

        frame = build_frame(img, item, {
            "i": i, "n": len(items),
            "label": labels[item["id"]] if labels[item["id"]] is not None else "null",
            "seed": seed[item["id"]] if seed[item["id"]] is not None else "null",
            "changed": labels[item["id"]] != seed[item["id"]],
            "verdict": review.get(item["id"]),
            "n_ok": sum(1 for x in items if review.get(x["id"]) == "ok"),
            "n_changed": sum(1 for x in items if review.get(x["id"]) == "changed"),
            "n_todo": sum(1 for x in items if x["id"] not in review),
        })
        cv2.imshow("labels", frame)
        k = cv2.waitKey(0) & 0xFF

        def advance():
            nonlocal i
            i = min(i + 1, len(items) - 1)

        if k in (ord("q"), 27):
            if dirty:
                save()
            break
        elif k == ord("s"):
            save()
        elif k == ord("a"):
            review[item["id"]] = "ok"
            dirty = True
            advance()
        elif ord("1") <= k <= ord("8") or k == ord("0"):
            new = None if k == ord("0") else CATEGORIES[k - ord("1")]
            labels[item["id"]] = new
            review[item["id"]] = "changed" if new != seed[item["id"]] else "ok"
            last_set = new
            dirty = True
            advance()
        elif k == ord("A") and last_set is not None:
            n = 0
            for x in items:
                if x["image_path"] == item["image_path"]:
                    labels[x["id"]] = last_set
                    review[x["id"]] = "changed" if last_set != seed[x["id"]] else "ok"
                    n += 1
            dirty = True
            print(f"{last_set} aplicado a {n} tarefas de {item['image_path']}")
            advance()
        elif k == ord("u"):
            labels[item["id"]] = seed[item["id"]]
            review.pop(item["id"], None)
            dirty = True
        elif k in (ord(" "), ord("n"), 83):
            advance()
        elif k in (ord("b"), 81):
            i = max(i - 1, 0)

    cv2.destroyAllWindows()
    changed = {k: (seed[k], labels[k]) for k in labels if labels[k] != seed[k]}
    print(f"\n{sum(1 for x in items if review.get(x['id']) == 'ok')} aprovadas, "
          f"{len(changed)} alteradas, "
          f"{sum(1 for x in items if x['id'] not in review)} sem veredito")
    for k, (a, b) in sorted(changed.items()):
        print(f"  {k}: {a} -> {b}")


if __name__ == "__main__":
    main()
