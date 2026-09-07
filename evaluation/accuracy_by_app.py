"""Break a run's predictions down by application.

OSWorld-G groups its items by GUI element type (see benchmark/buckets.json), not
by the application on screen, and the annotations carry no application field.
`benchmark/task_labels.json` supplies one per task. It is a hand-reviewed file:
every one of the 564 tasks was looked at with its instruction and its target box
drawn on the screenshot, which is the only way to get the ones where the target
sits in a different window from the one the screenshot is "about". The labels are
the eight OSWorld domains -- Chrome, GIMP, LibreOffice Impress / Calc / Writer,
VsCode, VLC and OS -- and null for tasks outside that set, which includes every
refusal task, since no application on screen answers those.

    python accuracy_by_app.py /results/dense_preds.json
    python accuracy_by_app.py /results/dense_preds.json /results/pruned_preds.json
"""
import argparse
import collections
import json
import os

# Tasks outside the eight categories; reported, never counted in the total.
NULL = "null (fora das 8)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+",
                    help="One or two *_preds.json files written by holo_osworld_g.py.")
    ap.add_argument("--labels", default="../benchmark/task_labels.json",
                    help="Per-task labels, as written by validate_labels.py.")
    ap.add_argument("--include_null", action="store_true",
                    help="Count the tasks labelled null in the total as well. They "
                         "are outside the eight categories and include every "
                         "refusal task, so by default they are reported but not "
                         "counted.")
    args = ap.parse_args()

    labels = json.load(open(args.labels))

    tables, names = [], []
    for path in args.predictions:
        with open(path) as f:
            preds = json.load(f)["predictions"]
        t = collections.defaultdict(lambda: [0, 0])
        for p in preds:
            app = labels.get(p["data_id"]) or "null (fora das 8)"
            t[app][1] += 1
            t[app][0] += bool(p["correct"])
        tables.append(t)
        names.append(os.path.basename(path).replace("_preds.json", ""))

    apps = sorted({a for t in tables for a in t}, key=lambda a: -tables[0].get(a, [0, 0])[1])

    def counted(table):
        """Sum over the eight categories; null is reported, never counted."""
        return (sum(v[0] for a, v in table.items() if args.include_null or a != NULL),
                sum(v[1] for a, v in table.items() if args.include_null or a != NULL))

    if len(tables) == 1:
        print(f"{'aplicação':28} {'acurácia':>9}  {'itens':>9}")
        for a in apps:
            if a == NULL:
                continue
            c, n = tables[0][a]
            print(f"{a:28} {100*c/n:8.1f}%  {c:4}/{n:<4}")
        c, n = counted(tables[0])
        print(f"{'TOTAL':28} {100*c/n:8.1f}%  {c:4}/{n:<4}")
        if NULL in tables[0]:
            c, n = tables[0][NULL]
            print(f"{'(null, nao contado)':28} {100*c/n:8.1f}%  {c:4}/{n:<4}")
        return

    a_name, b_name = names[0], names[1]
    print(f"{'aplicação':28} {a_name:>12} {b_name:>12} {'delta':>8}  itens")
    for a in apps:
        if a == NULL and not args.include_null:
            continue
        ca, na = tables[0].get(a, [0, 0])
        cb, nb = tables[1].get(a, [0, 0])
        if not na or not nb:
            continue
        pa, pb = 100 * ca / na, 100 * cb / nb
        print(f"{a:28} {pa:11.1f}% {pb:11.1f}% {pb-pa:+7.1f}  {na}")
    ca, na = counted(tables[0])
    cb, nb = counted(tables[1])
    print(f"{'TOTAL':28} {100*ca/na:11.1f}% {100*cb/nb:11.1f}% "
          f"{100*cb/nb - 100*ca/na:+7.1f}  {na}")


if __name__ == "__main__":
    main()
