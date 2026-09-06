"""The structured-extraction task set: messy documents with frozen ground truth.

This module is **domain content**, not engine code. It owns the documents an
agent is asked to read, the fields it is asked to pull out of them, and the
correct answer for every one of those fields. It imports nothing from the
engine; the engine reaches this domain only through the package-level
``get_suite()`` and ``get_evaluator()`` in :mod:`agent_engineer.domains.extraction`,
per ``agent_engineer/evaluation/INTERFACE.md``.

Everything here is data. Scoring lives in :mod:`agent_engineer.domains.extraction.evaluator`.

Each internal :class:`ExtractionTask` renders to a
:class:`~agent_engineer.evaluation.TaskSpec` via :meth:`ExtractionTask.to_task_spec`,
which is how the harness sees it: ``prompt`` becomes the task's prompt, and the
field specs and ground truth -- opaque to the harness -- travel in
``TaskSpec.metadata`` for the evaluator in this same package to read back.

The documents are deliberately realistic rather than tidy. Across the set you
will find inconsistent layouts (labelled key/value blocks, ASCII tables, and
running prose), OCR-style character noise (``0``/``O``, ``1``/``l``, ``5``/``S``,
split words, stray glyphs), fields that are simply absent and must be reported
as ``null`` rather than guessed, values stated in more than one place, and
plausible distractors -- a shipper address next to a consignee address, a
subtotal next to a total, a date of birth next to a date of loss.

Where a value appears twice, the document always makes one of them
authoritative in text a careful reader can act on ("supersedes", "REVISED",
"disregard"). Ground truth is therefore never a coin flip, which is what lets
the evaluator be deterministic.

Difficulty is banded by :class:`Difficulty` and spread on purpose. A weak
baseline agent should clear the ``EASY`` band and fail most of the ``HARD``
band: if everything passed there would be nothing for the loop to improve, and
if nothing passed the baseline score would be zero and the loop would have no
gradient to follow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_engineer.evaluation import DomainSuite, TaskSpec

__all__ = [
    "DOMAIN",
    "FieldKind",
    "Difficulty",
    "FieldSpec",
    "ExtractionTask",
    "TASK_SET",
    "TASKS_BY_ID",
    "get_task",
    "build_suite",
]

DOMAIN = "extraction"


class FieldKind(str, Enum):
    """How a field's value is compared. Drives normalization in the evaluator."""

    TEXT = "text"
    """Free text. Compared case-insensitively with whitespace collapsed."""

    MONEY = "money"
    """A monetary amount. Compared numerically, so '$1,234.50' == '1234.5'."""

    DATE = "date"
    """A calendar date. Compared as a date, so 'Jan 5, 2024' == '2024-01-05'."""

    INTEGER = "integer"
    """A whole number. Compared numerically, so '1,024' == 1024."""

    ENUM = "enum"
    """One of a closed set of values, compared like TEXT against `choices`."""

    LIST = "list"
    """An unordered collection of short strings, compared as a normalized set."""


class Difficulty(str, Enum):
    """Difficulty band. Used to range the task set, and to slice scores."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field an agent must extract, and what counts as getting it right."""

    name: str
    kind: FieldKind
    description: str
    choices: tuple[str, ...] = ()
    """For ENUM fields, the permitted values. Empty for every other kind."""

    def __post_init__(self) -> None:
        if self.kind is FieldKind.ENUM and not self.choices:
            raise ValueError(f"enum field {self.name!r} must declare choices")
        if self.kind is not FieldKind.ENUM and self.choices:
            raise ValueError(f"choices are only meaningful for enum fields, not {self.name!r}")


@dataclass(frozen=True, slots=True)
class ExtractionTask:
    """One document, the fields to pull from it, and the correct answer.

    ``prompt`` is the whole of what the agent is shown: the instruction, the
    field list, the required output shape, and the document itself. The engine
    hands ``prompt`` to a candidate agent and records the result as a
    ``Trajectory`` whose ``task_id`` matches :attr:`task_id`.

    A ground-truth value of ``None`` means the field is genuinely absent from
    the document and the agent is expected to say so rather than invent one.
    """

    task_id: str
    difficulty: Difficulty
    doc_type: str
    document: str
    fields: tuple[FieldSpec, ...]
    ground_truth: dict[str, Any]
    pass_threshold: float = 1.0
    """Fraction of fields that must be correct for the verdict to pass."""
    notes: str = ""
    """What makes this task hard. Read by humans, not by the evaluator."""

    _prompt: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"task {self.task_id!r} has duplicate field names")
        missing = sorted(set(names) - self.ground_truth.keys())
        if missing:
            raise ValueError(
                f"task {self.task_id!r} has no ground truth for: {', '.join(missing)}"
            )
        extra = sorted(self.ground_truth.keys() - set(names))
        if extra:
            raise ValueError(
                f"task {self.task_id!r} has ground truth for undeclared fields: "
                + ", ".join(extra)
            )
        if not 0.0 < self.pass_threshold <= 1.0:
            raise ValueError(f"task {self.task_id!r} has a pass_threshold outside (0, 1]")
        object.__setattr__(self, "_prompt", _build_prompt(self))

    @property
    def prompt(self) -> str:
        """The full instruction handed to the agent, document included."""
        return self._prompt

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields)

    def spec_for(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise KeyError(f"task {self.task_id!r} has no field {name!r}")

    def to_task_spec(self) -> TaskSpec:
        """Render as the harness sees it: prompt in the open, fixtures in metadata.

        ``metadata`` is opaque to the harness -- this is the domain's own
        wire format for it, and the evaluator in this package is the only
        other reader.
        """
        return TaskSpec(
            task_id=self.task_id,
            domain=DOMAIN,
            prompt=self.prompt,
            expected=None,
            metadata={
                "difficulty": self.difficulty.value,
                "doc_type": self.doc_type,
                "pass_threshold": self.pass_threshold,
                "fields": [
                    {
                        "name": spec.name,
                        "kind": spec.kind.value,
                        "description": spec.description,
                        "choices": list(spec.choices),
                    }
                    for spec in self.fields
                ],
                "ground_truth": dict(self.ground_truth),
            },
        )


def _build_prompt(task: ExtractionTask) -> str:
    """Render the instruction shown to the agent.

    The required output shape is stated explicitly and shown as a skeleton, so
    that an agent which returns the wrong shape has violated a contract it was
    given rather than one it had to guess.
    """
    lines = [
        f"Extract structured data from the {task.doc_type} below.",
        "",
        "Fields to extract:",
    ]
    for spec in task.fields:
        suffix = ""
        if spec.kind is FieldKind.ENUM:
            suffix = f" One of: {', '.join(spec.choices)}."
        lines.append(f"  - {spec.name} ({spec.kind.value}): {spec.description}{suffix}")
    skeleton = json.dumps({spec.name: None for spec in task.fields}, indent=2)
    lines += [
        "",
        "Rules:",
        "  - Respond with a single JSON object and nothing else.",
        "  - Use exactly these keys, with no extra keys and none omitted.",
        "  - If a field is genuinely absent from the document, use null.",
        "    Do not guess, and do not substitute a similar-looking value.",
        "  - Where a value appears more than once, the document says which one wins.",
        "",
        "Output shape:",
        skeleton,
        "",
        "--- BEGIN DOCUMENT ---",
        task.document,
        "--- END DOCUMENT ---",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# EASY -- clean, consistently labelled layouts.
#
# Every value sits on its own "Label: value" line, spelled the way the field is
# named, with no noise and no competing candidate. A naive agent that scans for
# a label and takes the rest of the line should clear this band. That is the
# point: it puts a floor under the baseline so the optimization loop starts
# from a non-zero score.
# --------------------------------------------------------------------------- #

_EASY_INVOICE = ExtractionTask(
    task_id="extract-easy-invoice-001",
    difficulty=Difficulty.EASY,
    doc_type="invoice",
    notes="Clean labelled key/value block. No noise, no distractors, no gaps.",
    document="""\
NORTHWIND SUPPLY CO.
1400 Alder Street, Portland, OR 97205

INVOICE

Invoice Number: INV-2024-00871
Invoice Date: 2024-03-14
Due Date: 2024-04-13
Customer Name: Beacon Hardware LLC
Purchase Order: PO-55219

Line Items
  Anodized brackets, 200 ct     840.00
  Stainless fasteners, 500 ct   310.00
  Freight                        45.50

Subtotal: 1195.50
Tax: 95.64
Total Due: 1291.14

Payment Terms: Net 30
""",
    fields=(
        FieldSpec("invoice_number", FieldKind.TEXT, "The invoice's own identifier."),
        FieldSpec("invoice_date", FieldKind.DATE, "The date the invoice was issued."),
        FieldSpec("customer_name", FieldKind.TEXT, "The organization being billed."),
        FieldSpec("total_due", FieldKind.MONEY, "The final amount payable."),
        FieldSpec("payment_terms", FieldKind.TEXT, "The stated payment terms."),
    ),
    ground_truth={
        "invoice_number": "INV-2024-00871",
        "invoice_date": "2024-03-14",
        "customer_name": "Beacon Hardware LLC",
        "total_due": "1291.14",
        "payment_terms": "Net 30",
    },
)

_EASY_PURCHASE_ORDER = ExtractionTask(
    task_id="extract-easy-purchase-order-002",
    difficulty=Difficulty.EASY,
    doc_type="purchase order",
    notes="Clean labels. Includes one integer field and one closed-set field.",
    document="""\
CEDAR RIDGE MANUFACTURING
PURCHASE ORDER

PO Number: PO-2024-1188
Order Date: 2024-05-02
Vendor Name: Halcyon Polymers Inc.
Ship To: Cedar Ridge Manufacturing, 88 Foundry Rd, Akron, OH 44301
Total Quantity: 1450
Order Status: Approved
Requested Delivery: 2024-05-30
Order Total: 22750.00
""",
    fields=(
        FieldSpec("po_number", FieldKind.TEXT, "The purchase order identifier."),
        FieldSpec("vendor_name", FieldKind.TEXT, "The supplier the order is placed with."),
        FieldSpec("order_date", FieldKind.DATE, "The date the order was placed."),
        FieldSpec("total_quantity", FieldKind.INTEGER, "Total units ordered."),
        FieldSpec("order_total", FieldKind.MONEY, "The total value of the order."),
        FieldSpec(
            "order_status",
            FieldKind.ENUM,
            "The order's current state.",
            choices=("Draft", "Approved", "Shipped", "Cancelled"),
        ),
    ),
    ground_truth={
        "po_number": "PO-2024-1188",
        "vendor_name": "Halcyon Polymers Inc.",
        "order_date": "2024-05-02",
        "total_quantity": 1450,
        "order_total": "22750.00",
        "order_status": "Approved",
    },
)

_EASY_LAB_REPORT = ExtractionTask(
    task_id="extract-easy-lab-report-003",
    difficulty=Difficulty.EASY,
    doc_type="laboratory report",
    notes="Clean labels with a short list field.",
    document="""\
MERIDIAN CLINICAL LABORATORIES
Specimen Report

Accession Number: ACC-90114-B
Patient Name: Dana R. Whitfield
Collection Date: 2024-01-22
Ordering Physician: Dr. Alan Reyes
Specimen Type: Whole blood

Panels Performed: Complete Blood Count, Basic Metabolic Panel, Lipid Panel

Result Status: Final
""",
    fields=(
        FieldSpec("accession_number", FieldKind.TEXT, "The lab's accession identifier."),
        FieldSpec("patient_name", FieldKind.TEXT, "The patient the specimen came from."),
        FieldSpec("collection_date", FieldKind.DATE, "The date the specimen was collected."),
        FieldSpec("ordering_physician", FieldKind.TEXT, "The physician who ordered the work."),
        FieldSpec("panels_performed", FieldKind.LIST, "The panels that were run."),
    ),
    ground_truth={
        "accession_number": "ACC-90114-B",
        "patient_name": "Dana R. Whitfield",
        "collection_date": "2024-01-22",
        "ordering_physician": "Dr. Alan Reyes",
        "panels_performed": [
            "Complete Blood Count",
            "Basic Metabolic Panel",
            "Lipid Panel",
        ],
    },
)


# --------------------------------------------------------------------------- #
# MEDIUM -- one complication each.
#
# Each task here breaks exactly one assumption the easy band rewarded: the
# label is not spelled like the field, or the value is in prose rather than on
# a labelled line, or the layout is a table, or the field is absent and must
# come back null. Solvable, but not by line-scanning alone.
# --------------------------------------------------------------------------- #

_MEDIUM_OCR_INVOICE = ExtractionTask(
    task_id="extract-medium-ocr-invoice-004",
    difficulty=Difficulty.MEDIUM,
    doc_type="scanned invoice (OCR output)",
    notes=(
        "OCR noise: O/0 and l/1 substitutions, split words, stray glyphs. "
        "The purchase order field is absent and must come back null."
    ),
    document="""\
QUARRY  P0INT  INDUSTRIAL   SUPPLY
    ~~ scanned copy - do not re-scan ~~

lnv0ice N0.  : QPI-00-4471
lnvoice Date : l2 February 2O24
B i l l   T o : Tamsin  Freightworks , Ltd.
Terms        : Net 45

  Descript1on                Amt
  ------------------------  --------
  Hydraul1c hose  (12m)       620.00
  Coupl1ngs, brass            185.25
  ------------------------  --------
  T0TAL DUE                   805.25
""",
    fields=(
        FieldSpec("invoice_number", FieldKind.TEXT, "The invoice's own identifier."),
        FieldSpec("invoice_date", FieldKind.DATE, "The date the invoice was issued."),
        FieldSpec("customer_name", FieldKind.TEXT, "The organization being billed."),
        FieldSpec("total_due", FieldKind.MONEY, "The final amount payable."),
        FieldSpec(
            "purchase_order_number",
            FieldKind.TEXT,
            "The customer's purchase order reference, if the document states one.",
        ),
    ),
    ground_truth={
        "invoice_number": "QPI-00-4471",
        "invoice_date": "2024-02-12",
        "customer_name": "Tamsin Freightworks, Ltd.",
        "total_due": "805.25",
        "purchase_order_number": None,
    },
)

_MEDIUM_PROSE_RECEIPT = ExtractionTask(
    task_id="extract-medium-prose-receipt-005",
    difficulty=Difficulty.MEDIUM,
    doc_type="expense reimbursement note",
    notes="Every value is in running prose. No labelled lines at all.",
    document="""\
Expense note, submitted by e-mail.

Hi Marguerite - attaching the write-up for the Denver trip. I flew out on the
morning of March 3rd, 2024 and the airfare came to eight hundred and twelve
dollars and forty cents ($812.40), booked through the corporate portal under
confirmation code TRV-4471-DEN. The hotel was three nights at the Wexler at
$189 a night, so $567 even before tax.

Total I am claiming is $1,379.40. Please code it to the Northeast Sales cost
centre, same as last quarter.

Thanks,
Ivo Petrakis
""",
    fields=(
        FieldSpec("claimant_name", FieldKind.TEXT, "The person submitting the claim."),
        FieldSpec("travel_date", FieldKind.DATE, "The date of outbound travel."),
        FieldSpec("confirmation_code", FieldKind.TEXT, "The booking confirmation code."),
        FieldSpec("airfare_amount", FieldKind.MONEY, "The cost of the airfare alone."),
        FieldSpec("total_claimed", FieldKind.MONEY, "The total amount being claimed."),
        FieldSpec("cost_centre", FieldKind.TEXT, "The cost centre to charge."),
    ),
    ground_truth={
        "claimant_name": "Ivo Petrakis",
        "travel_date": "2024-03-03",
        "confirmation_code": "TRV-4471-DEN",
        "airfare_amount": "812.40",
        "total_claimed": "1379.40",
        "cost_centre": "Northeast Sales",
    },
)

_MEDIUM_MANIFEST_TABLE = ExtractionTask(
    task_id="extract-medium-manifest-table-006",
    difficulty=Difficulty.MEDIUM,
    doc_type="shipping manifest",
    notes=(
        "Values live in an ASCII table, addressed by column rather than by label. "
        "Total pieces must be read off the totals row."
    ),
    document="""\
TRANSPACIFIC CONSOLIDATORS -- MANIFEST

Booking .......... BKG-77-30912
Vessel ........... MV Corvid Star
Sailing .......... 2024-06-11

+-------+---------------------------+--------+-----------+
| Crate | Contents                  | Pieces | Weight kg |
+-------+---------------------------+--------+-----------+
| C-01  | Ceramic insulators        |    120 |     418.0 |
| C-02  | Copper busbar, 3m         |     40 |     902.5 |
| C-03  | Assorted mounting hardware|    310 |     164.5 |
+-------+---------------------------+--------+-----------+
|       | TOTALS                    |    470 |    1485.0 |
+-------+---------------------------+--------+-----------+

Port of Discharge: Long Beach, CA
""",
    fields=(
        FieldSpec("booking_reference", FieldKind.TEXT, "The booking reference."),
        FieldSpec("vessel_name", FieldKind.TEXT, "The vessel carrying the shipment."),
        FieldSpec("sailing_date", FieldKind.DATE, "The sailing date."),
        FieldSpec("total_pieces", FieldKind.INTEGER, "Total pieces across all crates."),
        FieldSpec("port_of_discharge", FieldKind.TEXT, "Where the cargo is discharged."),
        FieldSpec("crate_ids", FieldKind.LIST, "The identifier of every crate listed."),
    ),
    ground_truth={
        "booking_reference": "BKG-77-30912",
        "vessel_name": "MV Corvid Star",
        "sailing_date": "2024-06-11",
        "total_pieces": 470,
        "port_of_discharge": "Long Beach, CA",
        "crate_ids": ["C-01", "C-02", "C-03"],
    },
)

_MEDIUM_CLAIM_DISTRACTOR = ExtractionTask(
    task_id="extract-medium-claim-distractor-007",
    difficulty=Difficulty.MEDIUM,
    doc_type="insurance claim intake form",
    notes=(
        "Three dates sit close together -- date of birth, date of loss, date "
        "reported -- and only one is asked for. The adjuster field is unassigned."
    ),
    document="""\
HARROWGATE MUTUAL -- FIRST NOTICE OF LOSS

Claim Ref:            CLM-2024-0553
Policy Number:        HM-PL-88214-C
Insured:              Priya Ramanathan
Date of Birth:        1981-07-19
Date of Loss:         2024-04-06
Date Reported:        2024-04-09
Loss Type:            Water damage
Estimated Loss Value: 14,200.00
Adjuster Assigned:    -- none yet --
""",
    fields=(
        FieldSpec("claim_reference", FieldKind.TEXT, "The claim reference number."),
        FieldSpec("policy_number", FieldKind.TEXT, "The policy number."),
        FieldSpec("insured_name", FieldKind.TEXT, "The insured party."),
        FieldSpec("date_of_loss", FieldKind.DATE, "The date the loss occurred."),
        FieldSpec("estimated_loss_value", FieldKind.MONEY, "The estimated value of the loss."),
        FieldSpec(
            "adjuster_name",
            FieldKind.TEXT,
            "The adjuster assigned to the claim, if one is named.",
        ),
    ),
    ground_truth={
        "claim_reference": "CLM-2024-0553",
        "policy_number": "HM-PL-88214-C",
        "insured_name": "Priya Ramanathan",
        "date_of_loss": "2024-04-06",
        "estimated_loss_value": "14200.00",
        "adjuster_name": None,
    },
)

_MEDIUM_RESUME = ExtractionTask(
    task_id="extract-medium-resume-008",
    difficulty=Difficulty.MEDIUM,
    doc_type="resume",
    notes=(
        "Free-form layout with a pipe-delimited skills list to normalize. No "
        "phone number is given, and years of experience is spelled out in prose."
    ),
    document="""\
                          MAEVE O'DONNELL
                     maeve.odonnell@example.net
                          Galway, Ireland

PROFILE
  Process engineer, eleven years in continuous-flow chemical manufacturing,
  the last four of them leading commissioning teams.

SKILLS
  Process simulation | HAZOP facilitation | P&ID review | Six Sigma (Black Belt)

EXPERIENCE
  Senior Process Engineer, Corrib Chemical Works       2019 - present
  Process Engineer, Anderson & Blake                   2013 - 2019

EDUCATION
  MEng Chemical Engineering, University of Galway, 2013
""",
    fields=(
        FieldSpec("candidate_name", FieldKind.TEXT, "The candidate's full name."),
        FieldSpec("email", FieldKind.TEXT, "The candidate's e-mail address."),
        FieldSpec("location", FieldKind.TEXT, "Where the candidate is based."),
        FieldSpec("years_experience", FieldKind.INTEGER, "Total years of experience stated."),
        FieldSpec("skills", FieldKind.LIST, "The listed skills."),
        FieldSpec(
            "phone_number",
            FieldKind.TEXT,
            "The candidate's phone number, if the resume gives one.",
        ),
    ),
    ground_truth={
        "candidate_name": "Maeve O'Donnell",
        "email": "maeve.odonnell@example.net",
        "location": "Galway, Ireland",
        "years_experience": 11,
        "skills": [
            "Process simulation",
            "HAZOP facilitation",
            "P&ID review",
            "Six Sigma (Black Belt)",
        ],
        "phone_number": None,
    },
)


# --------------------------------------------------------------------------- #
# HARD -- several complications at once.
#
# These stack noise on top of prose on top of contradiction. A value is stated
# twice and the second statement supersedes the first; the field asked for sits
# beside a more prominent one that is not it; the answer is split across
# sections and has to be assembled. A baseline agent is expected to fail most
# of this band, which is where the optimization loop gets its headroom.
# --------------------------------------------------------------------------- #

_HARD_SUPERSEDED_STATEMENT = ExtractionTask(
    task_id="extract-hard-superseded-statement-009",
    difficulty=Difficulty.HARD,
    doc_type="account statement with a correction notice",
    notes=(
        "The closing balance and the statement date are both printed twice. A "
        "correction notice further down supersedes the figures in the header, "
        "and the agent has to prefer the later, explicitly authoritative pair."
    ),
    document="""\
LOWTHER & FINCH -- CUSTODY ACCOUNT STATEMENT

Account Number:  LF-4402-9917
Account Holder:  Sandbourne Endowment Trust
Statement Date:  2024-07-31
Closing Balance: 1,204,880.15

  ... transaction detail omitted ...

*******************************************************************
CORRECTION NOTICE -- issued 2024-08-14

A late-settling trade was omitted from the figures printed at the top
of this statement. The corrected period end is 2024-08-02 and the
corrected closing balance is 1,198,455.60. These corrected figures
supersede the header block above and should be used for all
reconciliation purposes. Disregard the header figures entirely.
*******************************************************************

Statement prepared by: R. Considine, Custody Operations
""",
    fields=(
        FieldSpec("account_number", FieldKind.TEXT, "The account number."),
        FieldSpec("account_holder", FieldKind.TEXT, "The account holder."),
        FieldSpec(
            "statement_date",
            FieldKind.DATE,
            "The authoritative period-end date after any correction.",
        ),
        FieldSpec(
            "closing_balance",
            FieldKind.MONEY,
            "The authoritative closing balance after any correction.",
        ),
        FieldSpec("prepared_by", FieldKind.TEXT, "Who prepared the statement."),
    ),
    ground_truth={
        "account_number": "LF-4402-9917",
        "account_holder": "Sandbourne Endowment Trust",
        "statement_date": "2024-08-02",
        "closing_balance": "1198455.60",
        "prepared_by": "R. Considine",
    },
    pass_threshold=0.8,
)

_HARD_DISCHARGE_SUMMARY = ExtractionTask(
    task_id="extract-hard-discharge-summary-010",
    difficulty=Difficulty.HARD,
    doc_type="hospital discharge summary",
    notes=(
        "Dense clinical prose. The discharge medication list must be separated "
        "from the pre-admission list it sits beside, and the admitting "
        "diagnosis is not the discharge diagnosis."
    ),
    document="""\
ST BRENDAN'S GENERAL -- DISCHARGE SUMMARY

Patient: Colm Feeny            MRN: 771-2049
Admitted: 2024-02-17           Discharged: 2024-02-23

HISTORY
  Mr Feeny presented to the emergency department with pleuritic chest pain and
  a three-day history of productive cough. He was admitted under the working
  diagnosis of pulmonary embolism, which CT pulmonary angiography did not
  support. Further imaging and cultures established community-acquired
  pneumonia, and this is the diagnosis he is discharged with.

MEDICATIONS ON ADMISSION
  He was taking atorvastatin 20mg nightly and ramipril 5mg daily on arrival.

MEDICATIONS AT DISCHARGE
  Continue amoxicillin 500mg three times daily for a further five days.
  Continue ramipril 5mg daily. Atorvastatin has been stopped pending review.

FOLLOW-UP
  Respiratory outpatients in six weeks; a date has not yet been issued.
""",
    fields=(
        FieldSpec("patient_name", FieldKind.TEXT, "The patient's name."),
        FieldSpec("mrn", FieldKind.TEXT, "The medical record number."),
        FieldSpec("discharge_date", FieldKind.DATE, "The date of discharge."),
        FieldSpec(
            "discharge_diagnosis",
            FieldKind.TEXT,
            "The diagnosis the patient is discharged with.",
        ),
        FieldSpec(
            "discharge_medications",
            FieldKind.LIST,
            "The drug names the patient is to continue on discharge.",
        ),
        FieldSpec(
            "followup_appointment_date",
            FieldKind.DATE,
            "The scheduled follow-up date, if one has been issued.",
        ),
    ),
    ground_truth={
        "patient_name": "Colm Feeny",
        "mrn": "771-2049",
        "discharge_date": "2024-02-23",
        "discharge_diagnosis": "community-acquired pneumonia",
        "discharge_medications": ["amoxicillin", "ramipril"],
        "followup_appointment_date": None,
    },
    pass_threshold=0.8,
)

_HARD_CONTRACT_AMOUNT = ExtractionTask(
    task_id="extract-hard-contract-amount-011",
    difficulty=Difficulty.HARD,
    doc_type="services agreement extract",
    notes=(
        "The fee is given in words and in digits and the two disagree; the "
        "contract says words govern. The effective date is defined by a rule "
        "rather than stated, and must be resolved from the signature date."
    ),
    document="""\
SERVICES AGREEMENT (EXTRACT)

This Agreement is made between Ashgrove Analytics Limited ("the Supplier")
and Pellagio Foods NV ("the Client").

3.  CONTRACT SUM
    The Client shall pay the Supplier a fixed fee of eighty-seven thousand
    five hundred euro (EUR 78,500.00) exclusive of VAT. Where the amount
    expressed in words differs from the amount expressed in figures, the
    amount in words shall govern.

4.  TERM
    This Agreement takes effect on the thirtieth day following the date of
    last signature below, and continues for twelve months thereafter.

9.  GOVERNING LAW
    This Agreement is governed by the laws of Ireland.

SIGNED for and on behalf of the parties:
    Ashgrove Analytics Limited .......... 2024-09-04
    Pellagio Foods NV ................... 2024-09-11
""",
    fields=(
        FieldSpec("supplier_name", FieldKind.TEXT, "The supplier party."),
        FieldSpec("client_name", FieldKind.TEXT, "The client party."),
        FieldSpec("contract_sum", FieldKind.MONEY, "The governing fixed fee, excluding VAT."),
        FieldSpec("currency", FieldKind.TEXT, "The currency of the fee."),
        FieldSpec(
            "effective_date",
            FieldKind.DATE,
            "The date the agreement takes effect, resolved per its own terms.",
        ),
        FieldSpec("governing_law", FieldKind.TEXT, "The governing jurisdiction."),
    ),
    ground_truth={
        "supplier_name": "Ashgrove Analytics Limited",
        "client_name": "Pellagio Foods NV",
        "contract_sum": "87500.00",
        "currency": "EUR",
        "effective_date": "2024-10-11",
        "governing_law": "Ireland",
    },
    pass_threshold=0.8,
)

_HARD_FREIGHT_BILL = ExtractionTask(
    task_id="extract-hard-freight-bill-012",
    difficulty=Difficulty.HARD,
    doc_type="freight bill (OCR output)",
    notes=(
        "Shipper and consignee blocks are formatted identically and only the "
        "consignee is asked for. OCR noise throughout, and the amount payable "
        "sits below a larger and more prominent declared-value figure."
    ),
    document="""\
   PIKE  &  MERROW   FREIGHT      BlLL  0F  LADlNG
   ------------------------------------------------
   Pro Number  :  PM-8813-QQ
   Pickup      :  O3/l9/2O24

   SHlPPER                        C0NSlGNEE
   Kettleridge Mills, lnc.        Vantar Distribut1on GmbH
   2200 Millrace Ave              Hafenstrasse 4l
   Sandusky, 0H 44870             2O457  Hamburg

   Pieces .......  6
   Class ........  85
   Declared Value  45,000.00
   ------------------------------------------------
   Freight Charges .....   2,480.00
   Fuel Surcharge ......     372.00
   AM0UNT  PAYABLE .....   2,852.00

   Terms:  C0LLECT
""",
    fields=(
        FieldSpec("pro_number", FieldKind.TEXT, "The carrier's PRO number."),
        FieldSpec("pickup_date", FieldKind.DATE, "The pickup date."),
        FieldSpec("consignee_name", FieldKind.TEXT, "The consignee, not the shipper."),
        FieldSpec("consignee_city", FieldKind.TEXT, "The consignee's city."),
        FieldSpec("amount_payable", FieldKind.MONEY, "The amount actually payable."),
        FieldSpec(
            "freight_terms",
            FieldKind.ENUM,
            "Who pays the freight.",
            choices=("Prepaid", "Collect", "Third Party"),
        ),
    ),
    ground_truth={
        "pro_number": "PM-8813-QQ",
        "pickup_date": "2024-03-19",
        "consignee_name": "Vantar Distribution GmbH",
        "consignee_city": "Hamburg",
        "amount_payable": "2852.00",
        "freight_terms": "Collect",
    },
    pass_threshold=0.8,
)

_HARD_SPLIT_RFQ = ExtractionTask(
    task_id="extract-hard-split-rfq-013",
    difficulty=Difficulty.HARD,
    doc_type="request for quotation with amendments",
    notes=(
        "The answer is assembled across sections: an amendment moves the "
        "closing date, the quantity is the sum of two schedule lines stated "
        "as a total further down, and one requested field was struck out."
    ),
    document="""\
BOROUGH OF WESTMARCH -- REQUEST FOR QUOTATION

RFQ Reference: WM-RFQ-2024-041
Issued: 2024-05-06
Original Closing Date: 2024-05-27

SCHEDULE OF REQUIREMENTS
  Lot 1  Sodium hypochlorite, 25L drums ........  180 drums
  Lot 2  Sodium hypochlorite, 200L drums .......   40 drums
  Combined requirement across both lots .........  220 drums

AMENDMENT No. 1 -- issued 2024-05-15
  The closing date is extended to 2024-06-10. All other terms of the RFQ
  are unchanged. This amendment forms part of the RFQ.

AMENDMENT No. 2 -- issued 2024-05-21
  The requirement for a site visit is withdrawn. The site visit date
  previously given in Section 7 is struck out and no longer applies.

CONTACT
  Procurement enquiries: Wilhelmina Tarke, procurement@westmarch.example.gov
""",
    fields=(
        FieldSpec("rfq_reference", FieldKind.TEXT, "The RFQ reference."),
        FieldSpec(
            "closing_date",
            FieldKind.DATE,
            "The closing date in force after all amendments.",
        ),
        FieldSpec("total_drums", FieldKind.INTEGER, "Total drums required across all lots."),
        FieldSpec("contact_name", FieldKind.TEXT, "The named procurement contact."),
        FieldSpec("contact_email", FieldKind.TEXT, "The procurement contact e-mail."),
        FieldSpec(
            "site_visit_date",
            FieldKind.DATE,
            "The site visit date, if one still applies after all amendments.",
        ),
    ),
    ground_truth={
        "rfq_reference": "WM-RFQ-2024-041",
        "closing_date": "2024-06-10",
        "total_drums": 220,
        "contact_name": "Wilhelmina Tarke",
        "contact_email": "procurement@westmarch.example.gov",
        "site_visit_date": None,
    },
    pass_threshold=0.8,
)

_HARD_UTILITY_BILL = ExtractionTask(
    task_id="extract-hard-utility-bill-014",
    difficulty=Difficulty.HARD,
    doc_type="utility bill (OCR output)",
    notes=(
        "Heavy OCR noise. Three money figures compete -- previous balance, "
        "current charges, and total now due -- and the meter reading asked for "
        "is the current one, printed beside the previous one."
    ),
    document="""\
  THAMESIDE  ENERGY   ~  quarterly  statement
  ==========================================
  Acc0unt : TE-99-l40772
  Supply Address : Flat 6, 12 Corbridge Walk, Reading

  Bil1ing Peri0d : Ol Apr 2O24  to  3O Jun 2O24
  Bill Date      : O5 Jul 2O24

  METER  M-40l19
     Previ0us reading .....  28,114 kWh
     Current  reading .....  29,507 kWh
     Units used ...........   1,393 kWh

  Previous balance ..........   84.20
  Payment received - thank you  -84.20
  Current charges ...........  246.75
  ------------------------------------
  T0TAL  NOW  DUE ...........  246.75

  Payment due by 26 Jul 2O24.  Tariff: Fixed Saver 24
""",
    fields=(
        FieldSpec("account_number", FieldKind.TEXT, "The energy account number."),
        FieldSpec("bill_date", FieldKind.DATE, "The date the bill was issued."),
        FieldSpec("payment_due_date", FieldKind.DATE, "The date payment is due by."),
        FieldSpec("current_meter_reading", FieldKind.INTEGER, "The current meter reading in kWh."),
        FieldSpec("units_used", FieldKind.INTEGER, "Units consumed in the period, in kWh."),
        FieldSpec("total_now_due", FieldKind.MONEY, "The total amount now due."),
        FieldSpec("tariff_name", FieldKind.TEXT, "The tariff the account is on."),
    ),
    ground_truth={
        "account_number": "TE-99-140772",
        "bill_date": "2024-07-05",
        "payment_due_date": "2024-07-26",
        "current_meter_reading": 29507,
        "units_used": 1393,
        "total_now_due": "246.75",
        "tariff_name": "Fixed Saver 24",
    },
    pass_threshold=0.8,
)


# --------------------------------------------------------------------------- #
# The task set.
# --------------------------------------------------------------------------- #

TASK_SET: tuple[ExtractionTask, ...] = (
    _EASY_INVOICE,
    _EASY_PURCHASE_ORDER,
    _EASY_LAB_REPORT,
    _MEDIUM_OCR_INVOICE,
    _MEDIUM_PROSE_RECEIPT,
    _MEDIUM_MANIFEST_TABLE,
    _MEDIUM_CLAIM_DISTRACTOR,
    _MEDIUM_RESUME,
    _HARD_SUPERSEDED_STATEMENT,
    _HARD_DISCHARGE_SUMMARY,
    _HARD_CONTRACT_AMOUNT,
    _HARD_FREIGHT_BILL,
    _HARD_SPLIT_RFQ,
    _HARD_UTILITY_BILL,
)
"""The ordered structured-extraction task set. One of this domain's two exports."""

TASKS_BY_ID: dict[str, ExtractionTask] = {task.task_id: task for task in TASK_SET}

if len(TASKS_BY_ID) != len(TASK_SET):  # pragma: no cover - guarded by tests
    raise RuntimeError("TASK_SET contains duplicate task ids")


def get_task(task_id: str) -> ExtractionTask:
    """Look up a task by id, as tests and diagnostics do; the harness never needs this."""
    try:
        return TASKS_BY_ID[task_id]
    except KeyError:
        raise KeyError(
            f"unknown task_id {task_id!r}; the structured-extraction task set has "
            f"{len(TASK_SET)} tasks"
        ) from None


def build_suite() -> DomainSuite:
    """Render the whole task set as a :class:`DomainSuite` for the harness."""
    return DomainSuite(domain=DOMAIN, tasks=tuple(task.to_task_spec() for task in TASK_SET))
