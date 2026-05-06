# English Test Data for SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection

- - -
Siehe unten für die deutsche Version.
- - -

### Type

Corpus, Dataset

### Authors

Dominik Schlechtweg, Haim Dubossarsky, Simon Hengchen, Barbara McGillivray, Nina Tahmasebi

### Description

This data collection contains the English test data for [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948):

- a lemmatized English text corpus pair (`corpus1/lemma/`, `corpus2/lemma/`)
- 37 lemmas (targets) which have been annotated for their lexical semantic change between the two corpora (`targets.txt`)
- the annotated binary change scores of the targets for subtask 1, and their annotated graded change scores for subtask 2 (`truth/`)

__Corpus 1__ (lemma version)

- based on: [CCOHA](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/ccoha/) / [COHA](https://www.english-corpora.org/coha/)
- language: English
- time covered: 1810-1860
- size: ~6 million tokens
- format: lemmatized, sentence length > 9 (before removal of punctuation), no punctuation, sentences randomly shuffled
- encoding: UTF-8
- note: targets have been concatenated with their broad POS tag ("target_pos"); sentences are split at replacement tokens (10 x "@") and replacement tokens are removed

__Corpus 2__ (lemma version)

- based on: [CCOHA](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/ccoha/) / [COHA](https://www.english-corpora.org/coha/)
- language: English
- time covered: 1960-2010
- size: ~6 million tokens
- format: lemmatized, sentence length > 9 (before removal of punctuation), no punctuation, sentences randomly shuffled
- encoding: UTF-8
- note: targets have been concatenated with their broad POS tag ("target_pos"); sentences are split at replacement tokens (10 x "@") and replacement tokens are removed

Besides the official lemma version of the corpora for SemEval-2020 Task 1 we also provide the raw token version (`corpus1/token/`, `corpus2/token/`). It contains the raw sentences in the same order as in the lemma version. Find more information on the data and SemEval-2020 Task 1 in the paper referenced below.

The creation of the data was supported by the CRETA center and the CLARIN-D grant funded by the German Ministry for Education and Research (BMBF).

### Reference

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi. 2020. [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://www.aclweb.org/anthology/2020.semeval-1.1/). SemEval@COLING2020.

Reem Alatrash, Dominik Schlechtweg, Jonas Kuhn, and Sabine Schulte im Walde. 2020. [CCOHA: Clean Corpus of Historical American English](https://www.aclweb.org/anthology/2020.lrec-1.859/). In Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC’20). European Language Resources Association (ELRA).

Mark Davies. 2012. [Expanding Horizons in Historical Linguistics with the 400-Million Word Corpus of Historical American English](https://www.euppublishing.com/doi/abs/10.3366/cor.2012.0024?journalCode=cor). Corpora, 7(2):121–157.

### Download

The resources are [freely available for education, research and other non-commercial purposes](https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip).

- - -

# Englische Testdaten für SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection

### Typ

Korpus, Datensatz

### Autoren

Dominik Schlechtweg, Haim Dubossarsky, Simon Hengchen, Barbara McGillivray, Nina Tahmasebi

### Beschreibung

Diese Datensammlung enthält die Testdaten für [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948):

- ein lemmatisiertes englisches Textkorpuspaar (`corpus1/lemma/`, `corpus2/lemma/`)
- 37 Lemmata (Targets), die bezüglich ihres Bedeutungswandels zwischen den beiden Korpora annotiert wurden (`targets.txt`)
- die annotierten binären Bedeutungswandelwerte der Targets für Subtask 1, und ihre annotierteren gradierten Bedeutungswandelwerte für Subtask 2 (`truth/`)

__Korpus 1__ (Lemma-Version)

- basiert auf: [CCOHA](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/ccoha/) / [COHA](https://www.english-corpora.org/coha/)
- Sprache: Englisch
- Zeitperiode: 1810-1860
- Größe: ~6 Millionen Token
- Format: lemmatisiert, Sätzlänge > 9 (vor Satzzeichenentfernung), keine Satzzeichen, Sätze wurden zufällig gemischt
- Textkodierung: UTF-8
- Hinweis: Targets wurden mit ihrem allgemeinen POS-Tag konkateniert ("target_pos"); Sätze wurden an Ersetzungs-Token (10 x "@") getrennt und Ersetzungs-Token wurden entfernt

__Korpus 2__ (Lemma-Version)

- basiert auf: [CCOHA](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/ccoha/) / [COHA](https://www.english-corpora.org/coha/)
- Sprache: Englisch
- Zeitperiode: 1960-2010
- Größe: ~6 Millionen Token
- Format: lemmatisiert, Sätzlänge > 9 (vor Satzzeichenentfernung), keine Satzzeichen, Sätze wurden zufällig gemischt
- Textkodierung: UTF-8
- Hinweis: Targets wurden mit ihrem allgemeinen POS-Tag konkateniert ("target_pos"); Sätze wurden an Ersetzungs-Token (10 x "@") getrennt und Ersetzungs-Token wurden entfernt

Neben der offiziellen Lemma-Version der Korpora für SemEval-2020 Task 1, stellen wir auch die unverarbeitete Token-Version zur Verfügung (`corpus1/token/`, `corpus2/token/`). Sie enthält die unverarbeiteten Sätze in derselben Reihenfolge wie in der Lemma-Version. Weitere Informationen zu den Daten und zu SemEval-2020 Task 1 finden Sie in dem unten zitierten Papier.

Die Erstellung der Daten wurde unterstützt durch das CRETA-Zentrum und das CLARIN-D-Projekt gefördert durch das Bundesministerium für Bildung und Forschung (BMBF).

### Referenz

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi. 2020. [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://www.aclweb.org/anthology/2020.semeval-1.1/). SemEval@COLING2020.

Reem Alatrash, Dominik Schlechtweg, Jonas Kuhn, and Sabine Schulte im Walde. 2020. [CCOHA: Clean Corpus of Historical American English](https://www.aclweb.org/anthology/2020.lrec-1.859/). In Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC’20). European Language Resources Association (ELRA).

Mark Davies. 2012. [Expanding Horizons in Historical Linguistics with the 400-Million Word Corpus of Historical American English](https://www.euppublishing.com/doi/abs/10.3366/cor.2012.0024?journalCode=cor). Corpora, 7(2):121–157.

### Download

Die Ressourcen sind [frei verfügbar für Lehre, Forschung sowie andere nicht-kommerzielle Zwecke](https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip).

- - -
