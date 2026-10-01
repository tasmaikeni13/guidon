# Pinned provenance and limits of pretrained exposure

The preparation uses FineWeb-Edu at `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
and FineMath at `e92b25a616738fe95dc186b64dfb19f9c8525594`. Both pinned cards
declare ODC-BY v1.0 and link Common Crawl's terms. Attribution must name the
dataset creators, revisions and source URLs; dataset licensing does not establish
that every underlying page has identical rights. Raw texts and token payloads are
private research artifacts rather than Git files. The immutable card and payload
hashes are recorded in `source-provenance-v1.json` and `candidate-catalog-v1.json`.
See the pinned [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)
and [FineMath card](https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md).

FineMath's upstream benchmark filter is useful provenance, but the present
preparation independently checks every registered reference and answer variant.
Its exclusion counts and measured capacities, rather than the dataset card's
nominal sizes, determine whether the reservations can be published. Source groups
are pinned PSL registrable domains, including PRIVATE suffixes. The shift cohort
means different source components from training, guidance and primary test; it
does not assert a date shift or a measured difference in topic distributions.

Pythia-160M-deduped at `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e` is an
Apache-2.0 model trained on the globally deduplicated Pile. The registered source
checkpoint is shared across all continued-pretraining arms. Its actual 162,322,944
parameters and source logits have been independently checked. Dataset deduplication
in that lineage does not establish that a newly crawled page, a public question,
or a solution was absent from the base model's training. See the pinned
[Pythia card](https://huggingface.co/EleutherAI/pythia-160m-deduped/blob/582159a2dfe3e712a8d47ae83dec95ae3bde8e7e/README.md)
and [Pythia paper](https://arxiv.org/abs/2304.01373).

The Pile includes broad web collections and specific providers. We conservatively
mark provider overlap with ArXiv, Stack Exchange, Wikipedia, GitHub, PubMed/NIH
and Project Gutenberg as possible historical exposure. A provider match is an
inference from source metadata, not proof that a particular document was present.
Other providers remain unverified because web reposting and the Pile's broad web
components prevent an absence claim. The full base membership corpus is not part
of this preparation; no exact historical absence certificate is claimed. Provider
definitions are linked in the pinned
[Pile construction record](https://github.com/EleutherAI/the-pile/blob/df97f8651ae3da658b19659b3ceaa6a34b0fc014/README.md)
and [Pile paper](https://arxiv.org/abs/2101.00027).

Accordingly, every reserved continued-pretraining source retains an
`unverified_base_exposure` qualification. All seven public tasks also have
unverified base exposure and cannot support a clean-generalization claim merely
because incremental training and guidance passed exclusion checks. Their later
results must be reported with this qualification or excluded from clean claims.
The primary prospective contrast is aggregate loss on freshly reserved source
components with shared initialization; it establishes an optimizer comparison
under that lineage, rather than erasing the lineage. A later failed confirmation
still requires a newly sealed evaluation reservation before redesign is confirmed.
