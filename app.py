import base64
import io
import json
import os
import uuid
import imaplib
import smtplib
from email import message_from_bytes
from email.message import EmailMessage
from datetime import datetime
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytz
import streamlit as st

# ---- OpenAI (chat + vision) ----
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Handled later


# ------------- CONFIG ------------- #
st.set_page_config(
    page_title="PowerDash Interview Scheduler",
    layout="wide"
)


# ------------- HELPERS ------------- #
def get_openai_client() -> Optional[Any]:
    """Create an OpenAI client using Streamlit secrets or environment variable."""
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
        return None

    if OpenAI is None:
        st.error("OpenAI Python SDK is not installed. Make sure 'openai' is in requirements.txt.")
        return None

    return OpenAI(api_key=api_key)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to a base64 PNG data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_slots_from_image(image: Image.Image) -> List[Dict[str, str]]:
    """
    Use GPT-4o Mini Vision to parse free/busy calendar images into slots.
    Expected JSON format:
    [
      {
        "date": "2025-12-03",
        "start": "09:00",
        "end": "09:30"
      },
      ...
    ]
    """
    client = get_openai_client()
    if not client:
        return []

    data_url = image_to_base64(image)

    system_prompt = (
        "You are an assistant that extracts interview availability slots from images of calendar free/busy views. "
        "Return ONLY valid JSON, no commentary, formatted as a list of objects with keys: "
        "\"date\" (YYYY-MM-DD), \"start\" (HH:MM in 24-hour format), \"end\" (HH:MM in 24-hour format). "
        "If you cannot find any slots, return an empty list []."
    )

    user_text = (
        "Extract all available interview slots from this image of a calendar free/busy view. "
        "Assume the local timezone is the same across all slots. "
        "Again: respond with ONLY JSON."
    )

    try:
        # Using legacy ChatCompletion-style API for broad compatibility
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        # Ensure we capture just JSON if model adds backticks
        if content.startswith("```"):
            content = content.strip("`")
            # remove possible language hint like json\n
            if "\n" in content:
                content = content.split("\n", 1)[1]

        slots = json.loads(content)
        # Basic validation
        valid_slots = []
        for s in slots:
            if all(k in s for k in ("date", "start", "end")):
                valid_slots.append(
                    {
                        "date": str(s["date"]),
                        "start": str(s["start"]),
                        "end": str(s["end"]),
                    }
                )
        return valid_slots
    except Exception as e:
        st.error(f"Error parsing slots with GPT-4o-mini vision: {e}")
        return []


def ensure_session_state():
    if "slots" not in st.session_state:
        st.session_state["slots"] = []
    if "email_log" not in st.session_state:
        st.session_state["email_log"] = []
    if "parsed_replies" not in st.session_state:
        st.session_state["parsed_replies"] = []
    if "selected_slot_index" not in st.session_state:
        st.session_state["selected_slot_index"] = None


def format_slot_label(slot: Dict[str, str], idx: int) -> str:
    return f"{idx + 1} – {slot['date']} {slot['start']}–{slot['end']}"


def build_scheduling_email(
    candidate_name: str,
    role: str,
    hiring_manager_name: str,
    recruiter_name: str,
    slots: List[Dict[str, str]],
) -> str:
    """Builds a warm, professional scheduling email offering numbered slots."""
    if not slots:
        return "No slots available. Please add availability first."

    slot_lines = []
    for i, s in enumerate(slots, start=1):
        slot_lines.append(f"{i}. {s['date']} at {s['start']}–{s['end']}")

    slot_text = "\n".join(slot_lines)

    email_body = f"""Hi {candidate_name},

Thank you again for your interest in the {role} opportunity with us.

We’d love to arrange your interview with {hiring_manager_name}. Please review the available time options below and reply to this email with the **number only** of your preferred option (for example: "2"):

{slot_text}

If none of these options work, simply reply with a note to let us know and we’ll be happy to suggest alternatives.

Best regards,
{recruiter_name}
Talent Acquisition
"""
    return email_body


def send_email_smtp(
    subject: str,
    body: str,
    to_emails: List[str],
    cc_emails: Optional[List[str]] = None,
    attachment: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send an email with optional ICS attachment using SMTP credentials from Streamlit secrets.

    Expected secrets:
    - smtp_server
    - smtp_port (optional, default 587)
    - smtp_username
    - smtp_password
    - smtp_from (optional, defaults to smtp_username)
    """
    required_keys = ["smtp_server", "smtp_username", "smtp_password"]
    for key in required_keys:
        if key not in st.secrets:
            st.error(f"Missing '{key}' in Streamlit secrets.")
            return False

    smtp_server = st.secrets["smtp_server"]
    smtp_port = int(st.secrets["smtp_port"]) if "smtp_port" in st.secrets else 587
    smtp_username = st.secrets["smtp_username"]
    smtp_password = st.secrets["smtp_password"]
    smtp_from = st.secrets.get("smtp_from", smtp_username)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join([e for e in to_emails if e])
    if cc_emails:
        msg["Cc"] = ", ".join([e for e in cc_emails if e])

    msg.set_content(body)

    if attachment:
        # attachment: {"filename": str, "content": str, "maintype": str, "subtype": str, "params": dict}
        filename = attachment.get("filename", "attachment")
        content = attachment.get("content", "")
        maintype = attachment.get("maintype", "text")
        subtype = attachment.get("subtype", "plain")
        params = attachment.get("params", {})

        msg.add_attachment(
            content.encode("utf-8"),
            maintype=maintype,
            subtype=subtype,
            filename=filename,
            params=params,
        )

    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"SMTP send failed: {e}")
        return False


def fetch_unread_emails_imap() -> List[Dict[str, Any]]:
    """
    Fetch unread emails from IMAP inbox using credentials from Streamlit secrets.

    Expected secrets:
    - imap_server
    - imap_port (optional, default 993)
    - imap_username
    - imap_password
    """
    required_keys = ["imap_server", "imap_username", "imap_password"]
    for key in required_keys:
        if key not in st.secrets:
            st.error(f"Missing '{key}' in Streamlit secrets.")
            return []

    imap_server = st.secrets["imap_server"]
    imap_port = int(st.secrets["imap_port"]) if "imap_port" in st.secrets else 993
    imap_username = st.secrets["imap_username"]
    imap_password = st.secrets["imap_password"]

    emails = []
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(imap_username, imap_password)
        mail.select("INBOX")

        typ, data = mail.search(None, "UNSEEN")
        if typ != "OK":
            st.error("Failed to search IMAP mailbox.")
            return []

        for num in data[0].split():
            typ, msg_data = mail.fetch(num, "(RFC822)")
            if typ != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = message_from_bytes(raw_email)

            subject = msg["subject"] or "(no subject)"
            from_ = msg["from"] or "(unknown sender)"

            # Extract simple text body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            body += part.get_payload(decode=True).decode(charset, errors="ignore")
                        except Exception:
                            continue
            else:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                except Exception:
                    body = msg.get_payload()

            emails.append(
                {
                    "from": from_,
                    "subject": subject,
                    "body": body,
                }
            )

        mail.logout()
    except Exception as e:
        st.error(f"IMAP error: {e}")
        return []

    return emails


def detect_slot_choice_from_text(text: str) -> Dict[str, Any]:
    """
    Lightweight NLP to detect a single slot number choice.
    Flags as unclear if:
    - No numbers found
    - Multiple distinct numbers found
    """
    import re

    numbers = re.findall(r"\b([1-9][0-9]?)\b", text)
    numbers = [int(n) for n in numbers]

    if not numbers:
        return {"status": "unclear", "reason": "no numbers detected", "choice": None}

    unique_numbers = sorted(set(numbers))
    if len(unique_numbers) > 1:
        return {"status": "unclear", "reason": "multiple numbers detected", "choice": unique_numbers}

    return {"status": "ok", "reason": "", "choice": unique_numbers[0]}


def generate_ics(
    start_dt_local: datetime,
    end_dt_local: datetime,
    timezone_str: str,
    subject: str,
    description: str,
    location: str,
    organizer_email: str,
    attendees: List[str],
) -> str:
    """
    Generate an ICS string with METHOD:REQUEST.
    Times are converted to UTC (Z) and timezone is included in header.
    """
    tz = pytz.timezone(timezone_str)
    start_local = tz.localize(start_dt_local)
    end_local = tz.localize(end_dt_local)

    start_utc = start_local.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)

    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uuid.uuid4()}@powerdashhr.com"

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%SZ")

    attendee_lines = ""
    for a in attendees:
        if not a:
            continue
        attendee_lines += f"ATTENDEE;CN={a};ROLE=REQ-PARTICIPANT:MAILTO:{a}\n"

    ics = f"""BEGIN:VCALENDAR
PRODID:-//PowerDash HR//Interview Scheduler//EN
VERSION:2.0
CALSCALE:GREGORIAN
METHOD:REQUEST
BEGIN:VEVENT
DTSTAMP:{dtstamp}
DTSTART:{fmt(start_utc)}
DTEND:{fmt(end_utc)}
SUMMARY:{subject}
DESCRIPTION:{description}
UID:{uid}
ORGANIZER;CN=Recruiter:MAILTO:{organizer_email}
LOCATION:{location}
{attendee_lines}STATUS:CONFIRMED
SEQUENCE:0
END:VEVENT
END:VCALENDAR
"""
    return ics.strip()


# ------------- UI ------------- #
ensure_session_state()

st.title("PowerDash Interview Scheduler")

tab1, tab2, tab3 = st.tabs(
    ["New Scheduling Request", "Scheduler Inbox", "Calendar Invites"]
)

# -------- TAB 1: NEW SCHEDULING REQUEST -------- #
with tab1:
    st.subheader("1. Upload Hiring Manager Availability")

    col_upload, col_slots = st.columns([1, 1.2])

    with col_upload:
        file = st.file_uploader(
            "Upload free/busy screenshot (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        parse_button = st.button("Parse Availability", type="primary", use_container_width=True)

    with col_slots:
        st.markdown("**Extracted Slots**")
        if st.session_state["slots"]:
            for idx, slot in enumerate(st.session_state["slots"]):
                st.write(format_slot_label(slot, idx))
        else:
            st.info("No slots extracted yet. Upload a calendar view and click 'Parse Availability'.")

    if parse_button and file is not None:
        try:
            images = []
            if file.type == "application/pdf":
                # Convert PDF pages to images
                with fitz.open(stream=file.read(), filetype="pdf") as doc:
                    for page_index in range(len(doc)):
                        page = doc.load_page(page_index)
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
            else:
                image = Image.open(file)
                images.append(image)

            all_slots = []
            with st.spinner("Extracting slots with GPT-4o-mini Vision..."):
                for img in images:
                    slots = parse_slots_from_image(img)
                    all_slots.extend(slots)

            st.session_state["slots"] = all_slots
            if all_slots:
                st.success(f"Extracted {len(all_slots)} slots.")
            else:
                st.warning("No slots extracted. Check the image and try again.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("---")
    st.subheader("2. Generate Candidate Scheduling Email")

    with st.form("email_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            candidate_name = st.text_input("Candidate Name")
            candidate_email = st.text_input("Candidate Email")
            role = st.text_input("Role Title")
            hiring_manager_name = st.text_input("Hiring Manager Name")
            hiring_manager_email = st.text_input("Hiring Manager Email")

        with col_right:
            recruiter_name = st.text_input("Recruiter Name")
            recruiter_email = st.text_input("Recruiter Email (from address if using personal mailbox)", value=st.secrets.get("smtp_from", ""))

            # Choose slots to offer
            st.markdown("**Slots to Offer (by index)**")
            if st.session_state["slots"]:
                offer_indices = st.multiselect(
                    "Select which slots to offer (by position)",
                    options=list(range(len(st.session_state["slots"]))),
                    default=list(range(len(st.session_state["slots"]))),
                    format_func=lambda i: format_slot_label(st.session_state["slots"][i], i),
                )
            else:
                offer_indices = []

        submitted = st.form_submit_button("Generate & Send Email", type="primary")

        if submitted:
            if not all([candidate_name, candidate_email, role, hiring_manager_name, hiring_manager_email, recruiter_name]):
                st.error("Please fill in all required fields.")
            elif not st.session_state["slots"]:
                st.error("No availability slots have been extracted.")
            elif not offer_indices:
                st.error("Please select at least one slot to offer.")
            else:
                slots_to_offer = [st.session_state["slots"][i] for i in offer_indices]
                email_body = build_scheduling_email(
                    candidate_name=candidate_name,
                    role=role,
                    hiring_manager_name=hiring_manager_name,
                    recruiter_name=recruiter_name,
                    slots=slots_to_offer,
                )
                subject = f"Interview availability – {role}"

                success = send_email_smtp(
                    subject=subject,
                    body=email_body,
                    to_emails=[candidate_email],
                    cc_emails=[recruiter_email, hiring_manager_email],
                )

                if success:
                    st.success("Scheduling email sent successfully.")
                    st.session_state["email_log"].append(
                        {
                            "candidate": candidate_name,
                            "candidate_email": candidate_email,
                            "role": role,
                            "subject": subject,
                            "body": email_body,
                        }
                    )
                else:
                    st.error("Failed to send scheduling email.")


# -------- TAB 2: SCHEDULER INBOX -------- #
with tab2:
    st.subheader("Monitor Scheduling Mailbox (IMAP)")

    st.write("This reads **unread** messages only and does **not** modify or delete any email.")

    if st.button("Fetch Unread Replies", type="primary"):
        emails = fetch_unread_emails_imap()
        if not emails:
            st.info("No unread emails found or failed to fetch.")
        else:
            parsed_results = []
            for em in emails:
                detection = detect_slot_choice_from_text(em["body"])
                parsed_results.append(
                    {
                        "from": em["from"],
                        "subject": em["subject"],
                        "body_preview": em["body"][:200].replace("\n", " ") + ("..." if len(em["body"]) > 200 else ""),
                        "status": detection["status"],
                        "reason": detection["reason"],
                        "choice": detection["choice"],
                    }
                )

            st.session_state["parsed_replies"] = parsed_results

    if st.session_state["parsed_replies"]:
        st.markdown("### Parsed Replies")
        for i, pr in enumerate(st.session_state["parsed_replies"]):
            with st.expander(f"{i+1}. {pr['from']} – {pr['subject']}"):
                st.write(f"**Status:** {pr['status']}")
                if pr["status"] == "ok":
                    st.write(f"**Detected choice number:** {pr['choice']}")
                else:
                    st.write(f"**Reason:** {pr['reason']}")
                st.write(f"**Body preview:** {pr['body_preview']}")
    else:
        st.info("No parsed replies yet.")


# -------- TAB 3: CALENDAR INVITES -------- #
with tab3:
    st.subheader("Generate & Send Calendar Invites (ICS)")

    if not st.session_state["slots"]:
        st.warning("No slots available. Please parse availability in Tab 1 first.")
    else:
        st.markdown("### Available Slots")
        slot_index = st.radio(
            "Select slot for invite",
            options=list(range(len(st.session_state["slots"]))),
            format_func=lambda i: format_slot_label(st.session_state["slots"][i], i),
            index=0,
        )
        st.session_state["selected_slot_index"] = slot_index
        selected_slot = st.session_state["slots"][slot_index]

        st.markdown("### Invite Details")

        col_left, col_right = st.columns(2)

        with col_left:
            candidate_email_ci = st.text_input("Candidate Email")
            hiring_manager_email_ci = st.text_input("Hiring Manager Email")
            recruiter_email_ci = st.text_input("Recruiter Email (optional CC)", value=st.secrets.get("smtp_from", ""))

            interview_type = st.selectbox("Interview Type", ["Teams", "Face-to-Face"])
            timezone_str = st.text_input("Timezone (IANA format)", value="Europe/London")

        with col_right:
            role_ci = st.text_input("Role Title")
            organizer_email_ci = st.text_input("Organizer/Scheduling Mailbox", value=st.secrets.get("smtp_from", "scheduling@powerdashhr.com"))

            teams_link = ""
            location = ""

            if interview_type == "Teams":
                teams_link = st.text_area("Teams Meeting Link / Instructions")
                location = "Microsoft Teams"
            else:
                location = st.text_input("On-site Location")
                teams_link = ""

            additional_notes = st.text_area("Additional Notes (optional)", "")

        generate_btn = st.button("Generate & Send Invite", type="primary")

        if generate_btn:
            if not all([candidate_email_ci, hiring_manager_email_ci, role_ci, organizer_email_ci]):
                st.error("Please fill in all mandatory invite fields.")
            else:
                try:
                    date_str = selected_slot["date"]
                    start_str = selected_slot["start"]
                    end_str = selected_slot["end"]

                    start_dt_local = datetime.strptime(date_str + " " + start_str, "%Y-%m-%d %H:%M")
                    end_dt_local = datetime.strptime(date_str + " " + end_str, "%Y-%m-%d %H:%M")

                    subject = f"{role_ci} Interview"
                    description_parts = [f"Interview for: {role_ci}"]
                    if interview_type == "Teams" and teams_link:
                        description_parts.append(f"Teams link: {teams_link}")
                    if additional_notes:
                        description_parts.append(f"Notes: {additional_notes}")

                    description = "\n".join(description_parts)

                    ics_content = generate_ics(
                        start_dt_local=start_dt_local,
                        end_dt_local=end_dt_local,
                        timezone_str=timezone_str,
                        subject=subject,
                        description=description,
                        location=location,
                        organizer_email=organizer_email_ci,
                        attendees=[candidate_email_ci, hiring_manager_email_ci, recruiter_email_ci],
                    )

                    # Download button
                    st.download_button(
                        label="Download ICS File",
                        data=ics_content,
                        file_name="interview_invite.ics",
                        mime="text/calendar",
                    )

                    # Send via SMTP
                    email_body = (
                        f"Please find attached the calendar invite for your interview.\n\n"
                        f"{description}\n\n"
                        f"Best regards,\nTalent Acquisition"
                    )

                    attachment = {
                        "filename": "interview_invite.ics",
                        "content": ics_content,
                        "maintype": "text",
                        "subtype": "calendar",
                        "params": {"method": "REQUEST", "name": "interview_invite.ics"},
                    }

                    success = send_email_smtp(
                        subject=subject,
                        body=email_body,
                        to_emails=[candidate_email_ci, hiring_manager_email_ci],
                        cc_emails=[recruiter_email_ci] if recruiter_email_ci else None,
                        attachment=attachment,
                    )

                    if success:
                        st.success("ICS invite sent successfully.")
                    else:
                        st.error("Failed to send ICS invite via email.")

                except Exception as e:
                    st.error(f"Error generating or sending ICS invite: {e}")
