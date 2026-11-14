"""Create the Week 3 Word report from week3_results.json."""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "week3_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "Week_3_Model_Evaluation_Report.docx"
NAVY = "18324B"
ORANGE = "E86625"
PALE = "F3F6F8"
GRAY = "596579"


def shade(cell, color):
    properties = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:fill"), color)
    properties.append(fill)


def set_cell_text(cell, text, *, bold=False, color=NAVY, size=9):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def add_table(document, headers, data, widths=None, font_size=8.8):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    if widths:
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
    header = table.rows[0]
    set_repeat_table_header(header)
    for i, title in enumerate(headers):
        set_cell_text(header.cells[i], title, bold=True, color="FFFFFF", size=font_size)
        shade(header.cells[i], NAVY)
    for row_number, row in enumerate(data):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=font_size)
            if row_number % 2 == 1:
                shade(cells[i], PALE)
        tr_pr = table.rows[-1]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if widths:
            for cell, width in zip(cells, widths):
                cell.width = Inches(width)
    document.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_bullet(document, text):
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.add_run(text)


def main():
    d = Document()
    section = d.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = d.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for name, size, color in (("Heading 1", 16, NAVY), ("Heading 2", 11.5, ORANGE)):
        style = d.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(11 if name == "Heading 1" else 7)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True

    kicker = d.add_paragraph()
    kicker.paragraph_format.space_after = Pt(4)
    run = kicker.add_run("WEEK 3  |  MACHINE LEARNING")
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(ORANGE)

    title = d.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(3)
    title_run = title.add_run("Unsupervised Learning and Model Evaluation")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(23)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor.from_string(NAVY)
    subtitle = d.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    subtitle_run = subtitle.add_run("Customer segmentation and binary classification")
    subtitle_run.font.size = Pt(12)
    subtitle_run.font.color.rgb = RGBColor.from_string(GRAY)

    clustering = RESULTS["clustering"]
    pca = RESULTS["pca"]
    cv = RESULTS["cross_validation"]
    metrics = RESULTS["test_metrics"]
    cm = metrics["confusion_matrix"]
    d.add_heading("Executive summary", level=1)
    d.add_paragraph(
        f"This exercise applies K-Means and Ward hierarchical clustering to a reproducible dataset of {RESULTS['rows']} synthetic customer profiles, then evaluates a logistic regression classifier on a held-out test set. Four clusters gave the strongest K-Means silhouette score ({clustering['selected_silhouette']:.3f}) and closely recovered the generated segment structure (adjusted Rand index {clustering['selected_adjusted_rand_index']:.3f}). Two principal components retained {pca['cumulative_explained_variance']:.1%} of feature variance. The tuned classifier achieved {metrics['accuracy']:.1%} test accuracy and {metrics['roc_auc']:.3f} ROC-AUC. These figures demonstrate the workflow; because the data are synthetic, they are not evidence about real customers."
    )

    d.add_heading("Dataset and approach", level=1)
    d.add_paragraph(
        "The dataset contains 320 generated profiles with monthly spend, visit days per month, basket item count, and customer tenure. It includes four known generating segments and a separate binary high-value label. The segment names are retained only to check clustering quality; neither segment names nor the high-value label is supplied to K-Means, Ward clustering, or PCA. The high-value label is used only for the supervised classification task."
    )
    d.add_paragraph(
        "Numeric features were standardized before distance-based clustering. K-Means was compared for k values from 2 through 6 using inertia, silhouette, and adjusted Rand index. Ward hierarchical clustering was cut to four clusters for comparison. PCA was applied to the standardized four-feature matrix. For classification, a stratified 25% holdout was reserved before tuning. Logistic regression regularization was selected by stratified five-fold cross-validation on the remaining 240 training rows; the final metrics below use the untouched 80-row test set."
    )

    d.add_heading("Clustering results", level=1)
    d.add_paragraph(
        "The silhouette score measures how close each point is to its own cluster compared with the nearest alternative cluster; higher values indicate better separation. Inertia measures within-cluster squared distance and always decreases as k grows, so it is read with the silhouette and the diminishing improvement after four clusters. Adjusted Rand index compares the cluster assignments with the known generated segment labels while correcting for chance."
    )
    cluster_rows = [
        [str(item["k"]), f"{item['inertia']:.1f}", f"{item['silhouette']:.3f}", f"{item['adjusted_rand_index']:.3f}"]
        for item in clustering["kmeans"]
    ]
    add_table(d, ["K-Means k", "Inertia", "Silhouette", "Adjusted Rand"], cluster_rows, [1.15, 1.2, 1.35, 1.45])
    d.add_paragraph(
        f"Four clusters were selected: they had the best silhouette in the tested range ({clustering['selected_silhouette']:.3f}), and the ARI of {clustering['selected_adjusted_rand_index']:.3f} indicates strong agreement with the generated segment structure. Ward clustering at k = 4 produced a silhouette of {clustering['ward_silhouette']:.3f} and ARI of {clustering['ward_adjusted_rand_index']:.3f}, very similar to K-Means."
    )
    d.add_paragraph(
        f"PCA reduced the four standardized variables to two components. Component 1 explained {pca['explained_variance_ratios'][0]:.1%} of variance and component 2 explained {pca['explained_variance_ratios'][1]:.1%}; together they retained {pca['cumulative_explained_variance']:.1%}. This is a compact two-dimensional summary with some information loss, rather than a replacement for the original feature values."
    )

    d.add_heading("Classification and model evaluation", level=1)
    d.add_paragraph(
        "Cross-validation estimates performance across several training and validation partitions. The regularization parameter C controls the strength of the penalty on large model coefficients. The setting with the highest mean validation F1 was selected; ROC-AUC was also tracked to assess ranking across classification thresholds."
    )
    cv_rows = [[f"{item['C']:g}", f"{item['mean_f1']:.3f}", f"{item['mean_roc_auc']:.3f}"] for item in cv["grid"]]
    add_table(d, ["Regularization C", "Mean CV F1", "Mean CV ROC-AUC"], cv_rows, [1.75, 1.55, 1.8])
    d.add_paragraph(f"The selected value was C = {cv['selected_C']:g}, with mean cross-validation F1 of {max(item['mean_f1'] for item in cv['grid']):.3f}. The model was then fit to the full training portion and scored once on the holdout.")

    d.add_heading("Held-out test results", level=2)
    add_table(
        d,
        ["Actual class", "Predicted 0", "Predicted 1"],
        [["0", str(cm["tn"]), str(cm["fp"])], ["1", str(cm["fn"]), str(cm["tp"]) ]],
        [1.7, 1.7, 1.7],
    )
    add_table(
        d,
        ["Metric", "Result", "Interpretation"],
        [
            ["Accuracy", f"{metrics['accuracy']:.1%}", "Correct predictions across both classes"],
            ["Precision", f"{metrics['precision']:.1%}", "Predicted positives that were positive"],
            ["Recall", f"{metrics['recall']:.1%}", "Actual positives identified"],
            ["F1-score", f"{metrics['f1']:.3f}", "Harmonic mean of precision and recall"],
            ["ROC-AUC", f"{metrics['roc_auc']:.3f}", "Ranking quality across thresholds"],
            ["Specificity", f"{metrics['specificity']:.1%}", "Actual negatives identified"],
        ],
        [1.15, 1.0, 4.0],
        font_size=8.4,
    )
    d.add_paragraph(
        f"At the fixed 0.5 probability threshold, the classifier correctly identified {cm['tp']} of {cm['tp'] + cm['fn']} positive test cases and {cm['tn']} of {cm['tn'] + cm['fp']} negative cases. Its {metrics['roc_auc']:.3f} ROC-AUC shows useful ranking on this sample, while the 70% accuracy should be read alongside precision and recall. A majority-class prediction would score {max(RESULTS['holdout']['test_positive_rate'], 1 - RESULTS['holdout']['test_positive_rate']):.1%} accuracy on this balanced test split."
    )

    d.add_heading("Conclusion and limitations", level=1)
    d.add_paragraph(
        "On this generated example, both clustering methods recovered four well-separated groups, and PCA compressed the feature space while retaining most of its variance. Cross-validation selected a moderately regularized logistic regression model; the held-out metrics show a workable but imperfect classifier. The confusion matrix makes the remaining false positives and false negatives visible, while precision and recall summarize their different costs."
    )
    d.add_paragraph(
        "The dataset is synthetic, the sample is modest, and its target was generated from the same measured features. The unusually strong agreement between clustering and the stored segment labels reflects how the example was constructed. Real customer data may contain overlapping groups, missing values, privacy constraints, and shifting behavior. A real deployment would require domain review, a meaningful baseline, representative data, leakage checks, and monitoring."
    )
    d.add_paragraph(
        "The accompanying project includes the reproducible analysis script, generated CSV data, and machine-readable results. Run the analysis before rebuilding this report so its tables reflect the current results file."
    )

    d.core_properties.title = "Week 3 Unsupervised Learning and Model Evaluation"
    d.core_properties.subject = "Clustering, PCA, cross-validation, and classification evaluation"
    d.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
