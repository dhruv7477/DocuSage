"""
tests/eval/decline/make_resume_pdf.py
-------------------------------------
Builds the two synthetic resumes used by the decline eval. The people, the
companies, the schools and the addresses are invented; emails use
example.com.

- resume_talia_brennerholt.pdf: a cover letter on page 1 and a resume on
  page 2. The name is set in the largest font at the top of each page, so
  the PDF parser turns it into a heading, and the section titles use a
  middle size, so they become second-level headings. The name is also in
  the letter's signature.
- resume_rosalind_kettleborne.pdf: a three-page resume whose name is ONLY
  its page 1 title (no signature, no name in the email), with pages 2 and 3
  of sections and sub-sections ("Hackathons" > one heading per event). Every
  later chunk inherits the name in its heading path although no later page
  states it, and one fact ("First place, Riverbend Data Jam 2024") is
  stated only by a sub-heading. It has more chunks than the answer context
  holds, so page 1 can be left out.

Run it where PyMuPDF is installed (the api image has it):

    python tests/eval/decline/make_resume_pdf.py [output_dir]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).parent

NAME = "Talia Brennerholt"
CONTACT = "Portland, Oregon | talia.brennerholt@example.com | (555) 010-0142"
NAME_2 = "Rosalind Kettleborne"

PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("letter")
MARGIN = 54.0
NAME_SIZE = 24.0
SECTION_SIZE = 13.0
SUBSECTION_SIZE = 11.5
BODY_SIZE = 10.5
SMALL_SIZE = 10.0
REGULAR, BOLD = "helv", "hebo"
LINE_HEIGHT = 1.25


@dataclass(frozen=True)
class Line:
    """One run of text in the layout: a paragraph, heading or bullet."""

    text: str
    size: float = BODY_SIZE
    font: str = REGULAR
    gap_after: float = 4.0
    indent: float = 0.0


COVER_LETTER: list[Line] = [
    Line(NAME, NAME_SIZE, BOLD, 6),
    Line(CONTACT, SMALL_SIZE, gap_after=18),
    Line("March 3, 2026", gap_after=12),
    Line("Hiring Committee", gap_after=0),
    Line("Corvantis Labs", gap_after=0),
    Line("1200 Example Avenue, Suite 400, Denver, CO 80202", gap_after=12),
    Line("Dear Hiring Committee,", gap_after=8),
    Line(
        "I am applying for the Senior Data Engineer role at Corvantis Labs, which "
        "I found on your careers page. Over the past seven years I have built "
        "batch and streaming data platforms for logistics and retail companies, "
        "and I would like to bring that experience to your analytics team.",
        gap_after=8,
    ),
    Line(
        "At Pellwick Freight I led the migration of a nightly batch warehouse to "
        "a streaming pipeline built on Apache Kafka and Apache Flink, which cut "
        "reporting delays from twelve hours to under five minutes. I also "
        "introduced data quality checks that reduced broken dashboards by 80 "
        "percent, and I mentor the junior engineers on my team.",
        gap_after=8,
    ),
    Line(
        "I have attached my resume, two letters of reference, and a copy of my "
        "AWS Certified Data Engineer certificate. I would welcome the chance to "
        "discuss how I could help Corvantis Labs deliver reliable data to its "
        "customers. I am available to start on May 1, 2026.",
        gap_after=14,
    ),
    Line("Sincerely,", gap_after=2),
    Line(NAME),
]

RESUME: list[Line] = [
    Line(NAME, NAME_SIZE, BOLD, 6),
    Line(f"Data Engineer | {CONTACT}", SMALL_SIZE, gap_after=14),
    Line("Professional Summary", SECTION_SIZE, BOLD, 4),
    Line(
        "Data engineer with seven years of experience designing data pipelines, "
        "warehouses and data quality tooling. Comfortable owning a platform end "
        "to end, from ingestion to the dashboards that business teams use.",
        gap_after=12,
    ),
    Line("Experience", SECTION_SIZE, BOLD, 4),
    Line(
        "Senior Data Engineer, Pellwick Freight, Seattle, WA (2022 to present)",
        font=BOLD,
    ),
    Line(
        "- Moved nightly batch loads to streaming pipelines on Kafka and Flink; "
        "reporting delay fell from 12 hours to under 5 minutes.",
        indent=12,
    ),
    Line(
        "- Built a data quality framework with Great Expectations that cut broken "
        "dashboards by 80 percent.",
        indent=12,
    ),
    Line(
        "- Mentored four junior engineers and ran the team's on-call rotation.",
        indent=12,
        gap_after=8,
    ),
    Line(
        "Data Engineer, Marrowgate Retail Group, Boise, ID (2019 to 2022)",
        font=BOLD,
    ),
    Line(
        "- Designed a Snowflake warehouse that replaced 40 spreadsheets used for "
        "weekly sales reporting.",
        indent=12,
    ),
    Line(
        "- Wrote Airflow DAGs for 120 daily jobs and kept them above a 99.5 "
        "percent success rate.",
        indent=12,
        gap_after=8,
    ),
    Line(
        "Junior Data Analyst, Fenwold County Health Department, Boise, ID "
        "(2018 to 2019)",
        font=BOLD,
    ),
    Line(
        "- Cleaned and reported vaccination data for 14 clinics.",
        indent=12,
        gap_after=12,
    ),
    Line("Skills", SECTION_SIZE, BOLD, 4),
    Line("Languages: Python, SQL, Scala"),
    Line(
        "Data tools: Apache Kafka, Apache Flink, Apache Airflow, dbt, Snowflake, "
        "PostgreSQL"
    ),
    Line("Cloud: AWS (S3, Glue, Redshift, Lambda), Terraform, Docker", gap_after=12),
    Line("Education", SECTION_SIZE, BOLD, 4),
    Line(
        "B.S. in Computer Science, University of Eastmoor, 2018 (graduated with "
        "honors)",
        gap_after=12,
    ),
    Line("Certifications", SECTION_SIZE, BOLD, 4),
    Line("AWS Certified Data Engineer, Associate (2024)"),
]

# The second resume: the name is only the page 1 title.
RESUME_2_PAGE_1: list[Line] = [
    Line(NAME_2, NAME_SIZE, BOLD, 6),
    Line(
        "Machine Learning Engineer | Austin, Texas | r.k.builds@example.com | "
        "(555) 010-0177",
        SMALL_SIZE,
        gap_after=14,
    ),
    Line("Summary", SECTION_SIZE, BOLD, 4),
    Line(
        "Machine learning engineer with six years of experience shipping ranking, "
        "forecasting and anomaly detection models to production. Focused on "
        "reliable training pipelines, careful offline evaluation and models that "
        "teams can maintain.",
        gap_after=12,
    ),
    Line("Experience", SECTION_SIZE, BOLD, 4),
    Line(
        "Machine Learning Engineer, Dunmore Logistics, Austin, TX (2021 to present)",
        font=BOLD,
    ),
    Line(
        "- Rebuilt demand forecasting on LightGBM; error (MAPE) fell from 18 to 11 "
        "percent across 300 depots.",
        indent=12,
    ),
    Line(
        "- Introduced a shared feature store that cut new model set-up from weeks "
        "to two days.",
        indent=12,
    ),
    Line(
        "- Ran model on-call for the routing team and wrote its incident playbook.",
        indent=12,
        gap_after=8,
    ),
    Line("Data Scientist, Halvern Insurance, Dallas, TX (2018 to 2021)", font=BOLD),
    Line(
        "- Shipped a claims triage model that routes 40 percent of claims without "
        "manual review.",
        indent=12,
    ),
    Line(
        "- Built anomaly detection for billing fraud that recovered 2.1 million "
        "dollars in its first year.",
        indent=12,
        gap_after=12,
    ),
    Line("Education", SECTION_SIZE, BOLD, 4),
    Line("M.S. in Statistics, Corrin State University, 2018"),
    Line("B.S. in Mathematics, Corrin State University, 2016"),
]

RESUME_2_PAGE_2: list[Line] = [
    Line("Hackathons", SECTION_SIZE, BOLD, 4),
    # A fact stated only by this heading: the body below does not repeat it.
    Line("First place, Riverbend Data Jam 2024", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Built a flood risk map from river gauge feeds and rainfall forecasts "
        "in 36 hours.",
        indent=12,
    ),
    Line(
        "- Team of four; led the model and the evaluation against past flood "
        "records.",
        indent=12,
        gap_after=8,
    ),
    Line("Finalist, Northgate Health Hack 2023", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Prototype that flags missed follow-up appointments from clinic "
        "schedules.",
        indent=12,
    ),
    Line(
        "- Presented to a panel of clinic managers and two hospital IT leads.",
        indent=12,
        gap_after=8,
    ),
    Line("Participant, Civic Code Weekend 2022", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Cleaned and published three years of city bus delay data as an open "
        "dataset.",
        indent=12,
        gap_after=12,
    ),
    Line("Projects", SECTION_SIZE, BOLD, 4),
    Line("Open-source time series toolkit", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Maintainer of a small Python library for backtesting forecasts; 1,200 "
        "stars on GitHub.",
        indent=12,
        gap_after=8,
    ),
    Line("Model monitoring dashboard", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Tracks drift and calibration for 25 production models; alerts the "
        "owning team in Slack.",
        indent=12,
        gap_after=12,
    ),
    Line("Open data pipeline for transit agencies", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Nightly ingestion of schedule and real-time feeds for five transit "
        "agencies, with data quality reports.",
        indent=12,
    ),
]

RESUME_2_PAGE_3: list[Line] = [
    Line("Publications", SECTION_SIZE, BOLD, 4),
    Line("Calibrating demand forecasts for depot staffing", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Applied Forecasting Workshop, 2023. Shows how quantile forecasts cut "
        "overtime hours by 9 percent.",
        indent=12,
        gap_after=8,
    ),
    Line("Cheap baselines for claims triage", SUBSECTION_SIZE, BOLD, 2),
    Line(
        "- Insurance Analytics Forum, 2020. Compares gradient boosting with "
        "simple rules on 1.4 million claims.",
        indent=12,
        gap_after=12,
    ),
    Line("Talks", SECTION_SIZE, BOLD, 4),
    Line(
        "- Feature stores without the hype, Austin Data Meetup, 2023.",
        indent=12,
    ),
    Line(
        "- What we learned from a year of model on-call, MLOps Day Texas, 2024.",
        indent=12,
        gap_after=12,
    ),
    Line("Volunteering", SECTION_SIZE, BOLD, 4),
    Line(
        "- Mentor in a weekend statistics course for high school students, "
        "2019 to present.",
        indent=12,
        gap_after=12,
    ),
    Line("Certifications", SECTION_SIZE, BOLD, 4),
    Line("Google Cloud Professional Machine Learning Engineer (2023)"),
    Line("Certified Kubernetes Application Developer (2022)", gap_after=12),
    Line("Skills", SECTION_SIZE, BOLD, 4),
    Line("Languages: Python, SQL, Go"),
    Line("ML: PyTorch, scikit-learn, LightGBM, MLflow"),
    Line("Infrastructure: Kubernetes, Airflow, Google Cloud", gap_after=12),
    Line("Spoken languages", SECTION_SIZE, BOLD, 4),
    Line("English (native), Spanish (professional working proficiency)"),
]

# filename -> (author, PDF title, pages)
RESUMES: dict[str, tuple[str, str, list[list[Line]]]] = {
    "resume_talia_brennerholt.pdf": (
        NAME,
        f"{NAME} - Cover letter and resume",
        [COVER_LETTER, RESUME],
    ),
    "resume_rosalind_kettleborne.pdf": (
        NAME_2,
        f"{NAME_2} - Resume",
        [RESUME_2_PAGE_1, RESUME_2_PAGE_2, RESUME_2_PAGE_3],
    ),
}


def _draw_page(doc: fitz.Document, lines: list[Line]) -> None:
    """Lay out `lines` top to bottom on a new page, wrapping long text."""
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN
    for line in lines:
        x = MARGIN + line.indent
        for row in _wrap(line, PAGE_WIDTH - MARGIN - x):
            y += line.size * LINE_HEIGHT
            page.insert_text((x, y), row, fontsize=line.size, fontname=line.font)
        y += line.gap_after
        if y > PAGE_HEIGHT - MARGIN:
            raise RuntimeError("Page overflow: shorten the content.")


def _wrap(line: Line, width: float) -> list[str]:
    """Split `line.text` into rows no wider than `width` points."""
    rows: list[str] = []
    current = ""
    for word in line.text.split():
        candidate = f"{current} {word}".strip()
        too_wide = fitz.get_text_length(candidate, line.font, line.size) > width
        if too_wide and current:
            rows.append(current)
            current = word
        else:
            current = candidate
    rows.append(current)
    return rows


def build_resume(output: Path, name: str, title: str, pages: list[list[Line]]) -> Path:
    """Write one synthetic resume PDF to `output` and return it."""
    with fitz.open() as doc:
        for lines in pages:
            _draw_page(doc, lines)
        doc.set_metadata(
            {
                "title": title,
                "author": name,
                "subject": "Fictional resume for the DocuSage decline eval",
                "creator": "tests/eval/decline/make_resume_pdf.py",
                "producer": "PyMuPDF",
                "creationDate": "D:20260303000000Z",
                "modDate": "D:20260303000000Z",
            }
        )
        doc.save(output, garbage=4, deflate=True, no_new_id=True)
    return output


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    for filename, (person, pdf_title, page_lines) in RESUMES.items():
        print(build_resume(out_dir / filename, person, pdf_title, page_lines))
