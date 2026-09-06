"""Break a run's predictions down by application.

OSWorld-G groups its items by GUI element type (see benchmark/buckets.json), not
by the application on screen, and the annotations carry no application field.
`benchmark/app_labels.json` supplies one: every screenshot was labelled by
reading the application name out of the GNOME top bar in the image itself, so
these are read labels, not keywords guessed from the instruction text. The
labels are the eight OSWorld domains -- Chrome, GIMP, LibreOffice Impress /
Calc / Writer, VsCode, VLC and OS -- and null for the 62 items in applications
outside that set (Thunderbird, Evince, Gedit, Totem and friends).

    python accuracy_by_app.py /results/dense_preds.json
    python accuracy_by_app.py /results/dense_preds.json /results/pruned_preds.json
"""
import argparse
import collections
import json
import os


def load(preds_path, labels):
    with open(preds_path) as f:
        preds = json.load(f)["predictions"]
    rows = collections.defaultdict(lambda: [0, 0])  # app -> [correct, total]
    for p in preds:
        app = labels.get(p["image_path"]) or "null (fora das 8)"
        rows[app][1] += 1
        rows[app][0] += bool(p["correct"])
    return rows, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+",
                    help="One or two *_preds.json files written by holo_osworld_g.py.")
    ap.add_argument("--labels", default="../benchmark/app_labels.json")
    ap.add_argument("--exclude_refusal", action="store_true",
                    help="Drop the 54 refusal items, which no model answers.")
    args = ap.parse_args()

    labels = json.load(open(args.labels))

    tables, names = [], []
    for path in args.predictions:
        with open(path) as f:
            preds = json.load(f)["predictions"]
        if args.exclude_refusal:
            preds = [p for p in preds if p["box_type"] != "refusal"]
        t = collections.defaultdict(lambda: [0, 0])
        for p in preds:
            app = labels.get(p["image_path"]) or "null (fora das 8)"
            t[app][1] += 1
            t[app][0] += bool(p["correct"])
        tables.append(t)
        names.append(os.path.basename(path).replace("_preds.json", ""))

    apps = sorted({a for t in tables for a in t}, key=lambda a: -tables[0].get(a, [0, 0])[1])

    if len(tables) == 1:
        print(f"{'aplicação':28} {'acurácia':>9}  {'itens':>9}")
        for a in apps:
            c, n = tables[0][a]
            print(f"{a:28} {100*c/n:8.1f}%  {c:4}/{n:<4}")
        c = sum(v[0] for v in tables[0].values()); n = sum(v[1] for v in tables[0].values())
        print(f"{'TOTAL':28} {100*c/n:8.1f}%  {c:4}/{n:<4}")
        return

    a_name, b_name = names[0], names[1]
    print(f"{'aplicação':28} {a_name:>12} {b_name:>12} {'delta':>8}  itens")
    for a in apps:
        ca, na = tables[0].get(a, [0, 0])
        cb, nb = tables[1].get(a, [0, 0])
        if not na or not nb:
            continue
        pa, pb = 100 * ca / na, 100 * cb / nb
        print(f"{a:28} {pa:11.1f}% {pb:11.1f}% {pb-pa:+7.1f}  {na}")
    ca = sum(v[0] for v in tables[0].values()); na = sum(v[1] for v in tables[0].values())
    cb = sum(v[0] for v in tables[1].values()); nb = sum(v[1] for v in tables[1].values())
    print(f"{'TOTAL':28} {100*ca/na:11.1f}% {100*cb/nb:11.1f}% "
          f"{100*cb/nb - 100*ca/na:+7.1f}  {na}")


if __name__ == "__main__":
    main()
