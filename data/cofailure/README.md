# The co-failure panel

Per-item correctness for a purpose-built frontier, used by the section
"What the item level says" of the paper. The archived frontiers of this
repository record aggregate accuracies only and cannot test a claim about
how individual items fail; this panel can.

## Files

| file | what |
|---|---|
| `fmnist_indep_correct.npy` | bool, 10000 items x 21 models. `True` = the model got that test item right. Independent random-feature draw per width. |
| `fmnist_nested_correct.npy` | the same, with all widths sharing one random-feature draw (the adversarial control). |
| `fmnist_<family>_frontier.csv` | `D` (feature count), `x = ln D`, the chosen ridge penalty, accuracy, raw error, and the chance-corrected `eps`. |
| `fashion_mnist_raw/` | where `63_cofailure_panel.py` caches the four Fashion-MNIST `idx.gz` files (30 MB). It uses them if present and downloads them from the official repository if they are missing. They are omitted from this bundle for size; **for the Zenodo deposit, run `63` once and then include this folder**, which makes the archive need no network at all. |

## Reproducing

`python 63_cofailure_panel.py` rebuilds both `.npy` files from scratch in about
three minutes. It needs the network only on the first run, to fetch
Fashion-MNIST into `fashion_mnist_raw/`; every run after that is offline, as is
every other script. Every analysis in
`64_microfoundation.py` reads only the `.npy` and `.csv` files above, so the
measurement can be re-checked, and re-analysed differently, with no network
and no model training.

## Provenance

Fashion-MNIST, Xiao, Rasul and Vollgraf (2017), MIT licence,
`https://github.com/zalandoresearch/fashion-mnist`. Chosen over MNIST because
it saturates at a clearly non-zero error, so the frontier has a visible floor.
The models are random-feature ridge classifiers trained here; nothing about
them is taken from an upstream benchmark.
