"""
tests/eval/decline/make_resume_pdf.py
-------------------------------------
Builds the synthetic resume used by the decline eval: a cover letter on
page 1 and a resume on page 2. The person, the companies, the school and
the addresses are invented; emails use example.com.

The layout mirrors a real resume where it matters to the ingestion
pipeline: the name is set in the largest font at the top of each page, so
the PDF parser turns it into a heading, and the section titles use a
middle size, so they become second-level headings.

Run it where PyMuPDF is installed (the api image has it):

    python tests/eval/decline/make_resume_pdf.py [output.pdf]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

DEFAULT_OUTPUT = Path(__file__).with_name("resume_talia_brennerholt.pdf")

NAME = "Talia Brennerholt"
CONTACT = "Portland, Oregon | talia.brennerholt@example.com | (555) 010-0142"

PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("letter")
MARGIN = 54.0
NAME_SIZE = 24.0
SECTION_SIZE = 13.0
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


def build_resume(output: Path) -> Path:
    """Write the two-page synthetic resume PDF to `output` and return it."""
    with fitz.open() as doc:
        _draw_page(doc, COVER_LETTER)
        _draw_page(doc, RESUME)
        doc.set_metadata(
            {
                "title": f"{NAME} - Cover letter and resume",
                "author": NAME,
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
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    print(build_resume(target))
