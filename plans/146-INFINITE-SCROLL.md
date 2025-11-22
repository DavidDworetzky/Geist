# Infinite Scroll Implementation Plan

## Problem
The current chat implementation loads all chat history at once, which is inefficient for long conversations and increases initial load time.

## Proposed Solution
Implement infinite scroll for the chat history. 
1.  **Backend**: Add a new endpoint to fetch chat history in pages (chunks) with `offset` and `limit`.
2.  **Frontend**: Update the chat component to fetch the most recent messages first and load older messages as the user scrolls up.

## Implementation Details

### Backend
-   **File**: `app/models/database/chat_session.py`
    -   Add `get_paginated_chat_history(session_id: int, limit: int = 20, offset: int = 0) -> ChatHistory`.
    -   This function will load the JSON history, slice it `[-(offset+limit) : -offset]` (reversed for chat usually, or just slice normally and let frontend handle order. Since it's a list of messages appended over time, index 0 is oldest. Usually we want newest first for infinite scroll up.
    -   Actually, `ChatHistory` is a list. 0 is oldest.
    -   If we want infinite scroll *up*, we want the *last* N messages first.
    -   Let's define `offset` as "number of messages from the end" or just standard SQL-like offset from the beginning?
    -   Standard offset from beginning is hard if we don't know the total count without parsing.
    -   Better approach for JSON list:
        -   Load list.
        -   Total count = `len(history)`.
        -   Return slice.
        -   For "latest messages", we want `history[-limit:]`.
        -   Next page: `history[-(limit+offset) : -offset]`.
        -   Let's just stick to standard `skip/limit` (from start) or `last_n` logic?
        -   User asked for "paginated chat history with offsets".
        -   I'll implement `get_paginated_chat_history(session_id, offset, limit)` where `offset` is from the *start* of the list (oldest), effectively.
        -   Wait, if I use offset from start, how do I get the *latest* messages first without knowing the total?
        -   I'll return `total_count` in the response so frontend knows where to start fetching for "latest".
        -   OR, I can implement `offset` as "messages to skip from the *end*".
        -   Let's stick to standard: `offset` from start. Frontend can ask for `limit=20` and `offset=total - 20`.
        -   But frontend doesn't know `total`.
        -   Okay, I'll add a `get_chat_session_metadata` or similar? Or just return the total count in the paginated response.
        -   Simpler: `page` and `page_size`, where page 1 is the *latest* messages.
        -   Let's do: `get_chat_history(session_id, page=1, page_size=20)`.
        -   Page 1 = `history[-20:]`
        -   Page 2 = `history[-40:-20]`
        -   This seems friendlier for chat "scroll up".

-   **File**: `app/main.py`
    -   Add `GET /agent/chat_history/{session_id}/paginated`
    -   Query params: `page` (default 1), `page_size` (default 20).

### Frontend
-   **File**: `client/geist/src/Chat.tsx`
    -   On mount/id change: Fetch page 1.
    -   Update `ChatTextArea` or wrapper to detect scroll to top.
    -   On scroll top: Fetch page N+1.
    -   Prepend messages to `chatHistory`.

## Checklist
- [ ] Update `app/models/database/chat_session.py` with pagination logic.
- [ ] Add endpoint in `app/main.py`.
- [ ] Update `Chat.tsx` to use `useInfiniteScroll` or similar logic.

