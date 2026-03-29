import { useCallback, useRef, useState } from "react";
import "./App.css";

/**
 * Uses Vite proxy in dev: POST /chat -> http://127.0.0.1:8000/chat
 * Override with VITE_API_BASE if the API runs elsewhere.
 */
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function postChat(query) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Request failed");
  }
  return res.json();
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text:
        "Hi — I’m here as a friendly study buddy for help topics (streaks, subscriptions, your account, and more). " +
        "I’ll keep answers simple first, then go deeper if you need it, and I may ask what you want to do next. " +
        "This demo isn’t affiliated with Duolingo.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const data = await postChat(q);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: `Something went wrong: ${e.message}`,
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(scrollToBottom, 100);
    }
  }, [input, loading]);

  return (
    <div className="app">
      <header className="header">
        <h1>Help RAG chat</h1>
        <p className="sub">
          Independent demo · answers use retrieved public help snippets · not affiliated with Duolingo
        </p>
      </header>

      <main className="chat-panel">
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              <div className="bubble-inner">{msg.text}</div>
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <div className="sources">
                  <span className="sources-label">Sources</span>
                  <ul>
                    {msg.sources.map((s, j) => (
                      <li key={j}>
                        <a href={s.source_url} target="_blank" rel="noopener noreferrer">
                          {s.title || s.source_url}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="bubble assistant loading-row">
              <span className="typing" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span className="loading-text">Thinking…</span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <input
            type="text"
            placeholder="Ask about streaks, Super, refunds…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={loading}
            aria-label="Message"
          />
          <button type="button" onClick={send} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}
