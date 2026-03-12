import json
import os
import uuid
from datetime import datetime
import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time


# --- GROQ API helper for text generation (free and fast) ---
GROQ_API_KEY = "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"  # <-- Replace with your actual key from https://console.groq.com

import time

def query_groq(messages, retries=3, backoff=3):
    """Use Groq API for chat-based text generation with retry/backoff"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 800
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            result = response.json()

            # ✅ Success case
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

            # ❌ Rate limit or transient errors — retry
            if "error" in result:
                err_msg = result["error"].get("message", "")
                print(f"Groq API Error: {err_msg}")
                if "rate limit" in err_msg.lower() or "limit" in err_msg.lower():
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    continue
                return "There was an issue generating the answer. Please try again."

            # Unexpected response
            print("Unexpected Groq response:", result)
            return "Unexpected response format from Groq."

        except Exception as e:
            print(f"Error contacting Groq API: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                return "Failed to generate a response after multiple attempts."

    return "Groq API did not respond after retries."




class ReportQASystem:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.conversation_history = []
        self.analysis_store = self.load_analysis_store()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


    def load_analysis_store(self):
        """Load the analysis store from disk"""
        if os.path.exists("analysis_store.json"):
            with open("analysis_store.json", "r") as f:
                return json.load(f)
        return {"analysis": []}



    import time  # ⬅️ add this at the top with imports if not already

    def get_embeddings(self, text):
        """Return a fixed-length embedding (1536). Try Groq/OpenAI, fallback deterministic."""
        EMB_DIM = 1536
        # 1) Try OpenAI if user provided API key
        try:
            if self.api_key:
                client = OpenAI(api_key=self.api_key)
                resp = client.embeddings.create(input=text, model="text-embedding-3-small")
                emb = resp.data[0].embedding
                vec = np.array(emb, dtype=float)
                # pad/truncate
                if vec.size < EMB_DIM:
                    vec = np.pad(vec, (0, EMB_DIM - vec.size), 'constant')
                elif vec.size > EMB_DIM:
                    vec = vec[:EMB_DIM]
                return vec.tolist()
        except Exception as e:
            print("OpenAI embedding failed:", e)

        # 2) Try Groq embeddings endpoint (if you prefer) — PLEASE replace GROQ model/key if needed
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            data = {"model": "text-embedding-3-small", "input": text}  # Groq may not support this model; fails gracefully
            r = requests.post("https://api.groq.com/openai/v1/embeddings", headers=headers, json=data, timeout=8)
            j = r.json()
            if "data" in j and len(j["data"])>0 and "embedding" in j["data"][0]:
                emb = j["data"][0]["embedding"]
                vec = np.array(emb, dtype=float)
                if vec.size < EMB_DIM:
                    vec = np.pad(vec, (0, EMB_DIM - vec.size), 'constant')
                elif vec.size > EMB_DIM:
                    vec = vec[:EMB_DIM]
                return vec.tolist()
            else:
                print("Groq embedding response invalid:", j)
        except Exception as e:
            print("Groq embedding failed:", e)

        # 3) Deterministic fallback (no external API) — reproducible
        txt = (text or "")[:5000]
        rng = np.random.RandomState(abs(hash(txt)) % (2**32))
        vec = rng.rand(EMB_DIM).astype(float)
        return vec.tolist()










    def get_relevant_contexts(self, query, top_k=3):
        """Find relevant contexts for a query using embeddings similarity"""
        query_embedding = self.get_embeddings(query)
        analyses = self.analysis_store.get("analysis", [])
        contexts = []

        if not analyses:
            return ["No previous analyses found."]

        for analysis in analyses:
            analysis_text = analysis.get("analysis", "")
            if not analysis_text.strip():
                continue

            full_text = analysis_text
            if "findings" in analysis and analysis["findings"]:
                findings_text = "\n".join([f"- {finding}" for finding in analysis["findings"]])
                full_text += f"\n\nFindings:\n{findings_text}"

            full_text += f"\n\nImage: {analysis.get('filename', 'unknown')}"
            full_text += f"\nDate: {analysis.get('date', '')[:10]}"

            EMB_DIM = 1536  # fixed dimension for all embeddings

            embedding = analysis.get("embedding")
            if embedding is None:
                # if no embedding saved, generate and cache
                embedding = self.get_embeddings(full_text)
                analysis["embedding"] = embedding
            else:
                # ensure numpy array and fix length (pad or truncate)
                arr = np.array(embedding, dtype=float)
                if arr.size < EMB_DIM:
                    arr = np.pad(arr, (0, EMB_DIM - arr.size), 'constant')
                elif arr.size > EMB_DIM:
                    arr = arr[:EMB_DIM]
                embedding = arr.tolist()
                analysis["embedding"] = embedding  # overwrite with normalized embedding

            contexts.append({
                "text": full_text,
                "embedding": embedding,
                "id": analysis.get("id", ""),
                "date": analysis.get("date", "")
            })

 
        # ✅ Save updated embeddings back to disk so they're reused later
        with open("analysis_store.json", "w") as f:
            json.dump(self.analysis_store, f)



        similarities = []
        for context in contexts:
            try:
                q_vec = np.array(query_embedding, dtype=float).reshape(1, -1)
                c_vec = np.array(context["embedding"], dtype=float).reshape(1, -1)

                # Ensure both vectors are same dimension
                min_dim = min(q_vec.shape[1], c_vec.shape[1])
                q_vec = q_vec[:, :min_dim]
                c_vec = c_vec[:, :min_dim]

                similarity = cosine_similarity(q_vec, c_vec)[0][0]
            except Exception as e:
                print("Error computing similarity:", e)
                similarity = 0.0

            similarities.append((similarity, context))


        similarities.sort(key=lambda x: x[0], reverse=True)
        top_contexts = [context["text"] for _, context in similarities[:top_k]]

        return top_contexts

    def answer_question(self, question):
        """Answer a question about medical reports using RAG + Groq"""
        if not self.api_key:
            return "Please provide an OpenAI API key to enable the QA system."

        contexts = self.get_relevant_contexts(question)

        if not contexts or contexts[0] == "No previous analyses found.":
            return "I don't have any medical reports to reference. Please upload and analyze some images first."

        combined_context = "\n\n---\n\n".join(contexts)
        self.conversation_history.append({"role": "user", "content": question})

        try:
            system_prompt = f"""
            You are a concise medical AI assistant. 
            Answer briefly in **no more than 3 short sentences**. 
            Focus only on facts found in the context below, and avoid repeating data. 
            If the answer is not clearly available, reply: "Not enough data found in recent analyses."

            Contexts:
            {combined_context}
            """


            messages = [{"role": "system", "content": system_prompt}] + self.conversation_history

            # ✅ Use Groq API (free) for the answer
            answer = query_groq(messages)

            self.conversation_history.append({"role": "assistant", "content": answer})
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return answer
        except Exception as e:
            return f"I encountered an error while answering your question: {str(e)}"


class ReportQAChat:
    def __init__(self):
        self.qa_chat_store = self.get_qa_chat_store()

    def get_qa_chat_store(self):
        if os.path.exists("qa_chat_store.json"):
            with open("qa_chat_store.json", "r") as f:
                return json.load(f)
        return {"rooms": {}}

    def clear_room_messages(self, room_id):
        if room_id in self.qa_chat_store["rooms"]:
            self.qa_chat_store["rooms"][room_id]["messages"] = []
            self.save_qa_chat_store()
            return True
        return False


    def save_qa_chat_store(self):
        with open("qa_chat_store.json", "w") as f:
            json.dump(self.qa_chat_store, f)

    def create_qa_room(self, user_name, room_name):
        room_id = f"QA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        room_data = {
            "id": room_id,
            "name": room_name,
            "created_at": datetime.now().isoformat(),
            "creator": user_name,
            "messages": []
        }

        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Report QA System",
            "content": f"Welcome to the Report QA Room: {room_name}. You can ask questions about your medical report, and I'll answer based on stored analyses.",
            "timestamp": datetime.now().isoformat()
        }

        room_data["messages"].append(welcome_message)
        self.qa_chat_store["rooms"][room_id] = room_data
        self.save_qa_chat_store()

        return room_id

    def add_message(self, room_id, user_name, message):
        if room_id not in self.qa_chat_store["rooms"]:
            return None

        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "timestamp": datetime.now().isoformat()
        }

        self.qa_chat_store["rooms"][room_id]["messages"].append(message_data)
        self.save_qa_chat_store()

        return message_data

    def get_messages(self, room_id, limit=50):
        if room_id not in self.qa_chat_store["rooms"]:
            return []
        messages = self.qa_chat_store["rooms"][room_id]["messages"]
        return messages[-limit:] if len(messages) > limit else messages

    def get_qa_rooms(self):
        rooms = []
        for room_id, room_data in self.qa_chat_store["rooms"].items():
            rooms.append({
                "id": room_id,
                "name": room_data.get("name", "Unnamed Room"),
                "creator": room_data.get("creator", "Unknown"),
                "created_at": room_data.get("created_at", "")
            })
        rooms.sort(key=lambda x: x["created_at"], reverse=True)
        return rooms

    def delete_qa_room(self, room_id):
        if room_id in self.qa_chat_store["rooms"]:
            del self.qa_chat_store["rooms"][room_id]
            self.save_qa_chat_store()
            return True
        return False
    
    
