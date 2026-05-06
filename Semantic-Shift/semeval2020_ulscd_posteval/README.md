SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection
============================================================================================

**Authors**

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky, and Nina Tahmasebi

**Description**

This data collection contains the **post-evaluation data** for [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://languagechange.org/semeval):

-   the starting kit to download data, and examples for competing in the CodaLab challenge including baselines (`starting_kit/`),
-   the true binary change scores of the targets for Subtask 1, and their true graded change scores for Subtask 2 (`test_data_truth/`),
-   the scoring program used to score submissions against the true test data in the evaluation and post-evaluation phase (`scoring_program/`),
-   the results of the evaluation phase including
    -   the final rankings of the participating teams by their best submission (`results/rankings_teams.csv`),
    -   the submitted files of each team (`results/submissions/`),
    -   an overview of the results for each submission ordered by team (`results/submissions_results.csv`),
    -   analysis plots (`plots/`) displaying the results:
        -   under `per_target/` we provide the gold change scores and the normalized prediction error of target words plotted against their frequency and polysemy statistics,
        -   under `per_team/` we provide the model predictions from the best submission per team (per subtask) plotted against frequency/polysemy statistics and performance on gold data (gray lines give the correlation with the respective variable in the gold data); we also provide plots of visualizing the teams' prediction similarities.

Some remarks:

-   the paper referenced below remains the only source for the rankings between teams,
-   some teams were disqualified, and are thus removed from the analyses and the rankings present in the paper,
-   some teams have changed names, resulting in a discrepancy between team names under `results/` and team names in the paper. The paper contains a key to match old names with new names.

**Test Data** for SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection can be found using the links below:

-   [English](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/)
-   [German](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-ger/)
-   [Latin](https://zenodo.org/record/3734089)
-   [Swedish](https://zenodo.org/record/3730550)

Please find more information on the provided data in the paper referenced below.

**Reference**

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi. 2020. [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://languagechange.org/semeval). SemEval@COLING2020.

The resources are [freely available for education, research and other non-commercial purposes](https://zenodo.org/record/3931969).

```    @inproceedings{schlechtweg2020semeval,
    title = "{S}em{E}val-2020 {T}ask 1: {U}nsupervised {L}exical {S}emantic {C}hange {D}etection",
    author = "Schlechtweg, Dominik and McGillivray, Barbara and Hengchen, Simon and Dubossarsky, Haim and Tahmasebi, Nina",
    booktitle = "To appear in Proceedings of the 14th International Workshop on Semantic Evaluation",
    year = "2020",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics"}
```
 
