import streamlit as st
from datetime import datetime
import json
import os
import uuid
import time
from openai import OpenAI

# =========================
# STORAGE
# =========================

CHAT_FILE = "chat_store.json"


def get_chat_store():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return {"rooms": {}}


def save_chat_store(store):
    with open(CHAT_FILE, "w") as f:
        json.dump(store, f, indent=2)


# =========================
# CORE CHAT FUNCTIONS
# =========================

def create_chat_room(case_id, creator_name, case_description):
    store = get_chat_store()

    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [creator_name, "Dr. AI Assistant"],
            "messages": []
        }

        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Dr. AI Assistant",
            "content": f"Welcome doctors. This is a collaborative case discussion for: {case_description}",
            "type": "text",
            "timestamp": datetime.now().isoformat()
        }

        room_data["messages"].append(welcome_message)
        store["rooms"][case_id] = room_data
        save_chat_store(store)

    return case_id


def join_chat_room(case_id, user_name):
    store = get_chat_store()

    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    return False


def add_message(case_id, user_name, message, message_type="text"):
    store = get_chat_store()

    if case_id in store["rooms"]:
        store["rooms"][case_id]["messages"].append({
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        })
        save_chat_store(store)


def get_messages(case_id, limit=100):
    store = get_chat_store()
    if case_id in store["rooms"]:
        return store["rooms"][case_id]["messages"][-limit:]
    return []


def get_available_rooms():
    store = get_chat_store()
    rooms = []

    for room_id, data in store["rooms"].items():
        rooms.append({
            "id": room_id,
            "description": data["description"],
            "creator": data["creator"],
            "created_at": data["created_at"],
            "participants": len(data["participants"])
        })

    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms


# =========================
# AI RESPONSE
# =========================

def get_openai_response(user_question, case_description, findings=None, api_key=None):
    if not api_key:
        return "Please configure your OpenAI API key in the sidebar."

    client = OpenAI(api_key=api_key)

    findings_text = ""
    if findings:
        findings_text = "Key findings:\n" + "\n".join([f"- {f}" for f in findings])

    system_prompt = f"""
You are Dr. AI Assistant helping doctors collaborate on a medical case.

Case: {case_description}

{findings_text}

Reply in short, professional, clinical style.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=250,
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {e}"


# =========================
# STREAMLIT UI
# =========================

def render_chat_interface():
    st.subheader("🧑‍⚕️👩‍⚕️ Multi-Doctor Collaboration System")

    # ---------- DOCTOR SELECTOR ----------

    doctor_list = ["Dr. Sumit", "Dr. Meenal", "Dr. Aachal", "Dr. Guest"]
    user_name = st.selectbox("Select doctor", doctor_list)

    st.session_state.user_name = user_name

    tab1, tab2 = st.tabs(["Join Existing Case", "Create New Case"])

    # ---------- JOIN CASE ----------
    with tab1:
        rooms = get_available_rooms()
        if rooms:
            room_options = {f"{r['id']} - {r['description']}": r["id"] for r in rooms}
            selected = st.selectbox("Select Case", list(room_options.keys()))

            if st.button("Join Discussion"):
                cid = room_options[selected]
                join_chat_room(cid, user_name)
                st.session_state.current_case_id = cid
                st.rerun()
        else:
            st.info("No active discussions available.")

    # ---------- CREATE CASE ----------
    with tab2:
        desc = st.text_input("Case Description")
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if st.button("Create Discussion"):
            if desc.strip():
                cid = create_chat_room(case_id, user_name, desc)
                st.session_state.current_case_id = cid
                st.rerun()
            else:
                st.error("Please enter a case description.")

    # ---------- ACTIVE CHAT ----------
    if "current_case_id" in st.session_state:
        case_id = st.session_state.current_case_id
        store = get_chat_store()

        if case_id not in store["rooms"]:
            st.error("This case discussion no longer exists.")
            if st.button("Return to case list"):
                del st.session_state.current_case_id
                st.rerun()
            return

        room = store["rooms"][case_id]

        st.markdown(f"## 🗂 Case: {room['description']}")
        st.caption(f"Created by {room['creator']}")

        st.markdown("### 👥 Participants")
        st.info(" | ".join(room["participants"]))

        messages = get_messages(case_id)

        for msg in messages:
            avatar = "🤖" if msg["user"] == "Dr. AI Assistant" else "👨‍⚕️"
            with st.chat_message(msg["user"], avatar=avatar):
                st.write(msg["content"])

        message = st.chat_input("Type message to doctors or AI...")

        if message:
            add_message(case_id, user_name, message)

            api_key = st.session_state.get("openai_key")

            with st.spinner("Dr. AI Assistant thinking..."):
                time.sleep(0.6)
                reply = get_openai_response(message, room["description"], None, api_key)

            add_message(case_id, "Dr. AI Assistant", reply)
            st.rerun()


# =========================
# MANUAL ROOM CREATION
# =========================

def create_manual_chat_room(creator_name, case_description):
    case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return create_chat_room(case_id, creator_name, case_description)
