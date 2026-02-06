const DATA_URL = "data/articles.json";
const FORUM_URL = "data/forum.json";
const AGENTBOOK_FORUM_URL = "data/agentbook_forum.json";

const SECTION_ORDER = [
  {
    id: "spain",
    label: "Spain",
    subtitle: "National affairs, public services and daily life"
  },
  {
    id: "world",
    label: "World",
    subtitle: "International developments with local impact"
  },
  {
    id: "business",
    label: "Business",
    subtitle: "Markets, jobs, housing and cost of living"
  },
  {
    id: "politics",
    label: "Politics",
    subtitle: "Institutions, agreements and policy decisions"
  },
  {
    id: "technology",
    label: "Technology",
    subtitle: "AI, tech industry, devices and digital regulation"
  },
  {
    id: "opinion",
    label: "Opinion",
    subtitle: "Analysis and editorial perspectives"
  }
];

const escapeHtml = (text) => {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
};

const slug = (text) =>
  String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const formatDateShort = (iso) => {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short"
    });
  } catch {
    return iso;
  }
};

const formatDateLong = (dateObj = new Date()) => {
  try {
    return dateObj.toLocaleDateString("en-GB", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric"
    });
  } catch {
    return "";
  }
};

const formatDateTime = (iso) => {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch {
    return iso;
  }
};

const normalizeKey = (text) =>
  String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

const normalizeInlineArtifacts = (text) =>
  String(text || "")
    .replace(/\{[^{}]*(?:["']text["']|&#x27;text&#x27;|["']source["']|&#x27;source&#x27;)[^{}]*\}/gi, " ")
    .replace(/\(\s*https?:\/\/[^)]+\)/gi, " ")
    .replace(/\s+\[[0-9]{1,2}\]\s*$/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();

const isSourceCitationLine = (text) => {
  const raw = String(text || "").replace(/\s+/g, " ").trim();
  if (!raw) return false;
  if (/^\[\d{1,2}\]\s/.test(raw) && /(https?:\/\/| - |\s\(|\.\w{2,6}\b)/i.test(raw)) return true;
  if (/^(sources?|fuentes?)\b/i.test(raw)) return true;
  if (/https?:\/\/\S+/i.test(raw) && /\[\d{1,2}\]/.test(raw)) return true;
  return false;
};

const isMetaText = (text) => {
  const raw = String(text || "").replace(/\s+/g, " ").trim();
  if (!raw) return true;
  if (/^\{.*\}$/.test(raw) && /(?:text|source|certainty)/i.test(raw)) return true;
  if (isSourceCitationLine(raw)) return true;

  const key = normalizeKey(raw);
  if (!key) return true;
  if (/(i need to|the user wants|let me|first|second|third|next|finally)/.test(key)) return true;
  if (/(the title is|the section is|rules say|this section|now using|now structuring)/.test(key)) return true;
  if (/(the focus is|consulted sources|practical conclusion is clear|available evidence points)/.test(key)) return true;
  if (/(operationally the key is|following this topic requires continuous tracking)/.test(key)) return true;
  if (/(for readers in spain|main verified facts and immediate context|concrete effects on households)/.test(key)) return true;
  if (/(what comes next what to watch|threats opportunities and short term watchpoints)/.test(key)) return true;
  if (/(source\s*[:=]|certainty\s*[:=]|\bjson\b|\btext\b\s*[:=])/i.test(key)) return true;
  if (/(el usuario|voy a|debo|primero voy a|metacomentario|espanol de espana)/.test(key)) return true;
  if (/(^okay\b|^vale\b|^alternatively\b|^in summary\b)/.test(key)) return true;
  return false;
};

const cleanForumText = (text) => {
  if (!text) return "";
  return String(text)
    .replace(/<think[\s\S]*?<\/think>/gi, " ")
    .replace(/<analysis[\s\S]*?<\/analysis>/gi, " ")
    .replace(/&lt;think&gt;[\s\S]*?&lt;\/think&gt;/gi, " ")
    .replace(/&lt;analysis&gt;[\s\S]*?&lt;\/analysis&gt;/gi, " ")
    .replace(/\*\*/g, "")
    .replace(/^#+\s*/gm, "")
    .replace(/^\s*round\s+\d+\s*[:\-]\s*/gim, "")
    .replace(/^\s*editorial strategist'?s response[:\-]?\s*/gim, "")
    .replace(/^\s*fact checker'?s response[:\-]?\s*/gim, "")
    .replace(/^\s*summary[:\-]?\s*/gim, "")
    .replace(/\bno chain[- ]of[- ]thought\b/gi, "")
    .replace(/\bno bullet list\b/gi, "")
    .replace(/ +([,.;:!?])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
};

const cleanBodyHtml = (input) => {
  if (!input) return "";
  const raw = String(input)
    .replace(/<think[\s\S]*?<\/think>/gi, "")
    .replace(/<analysis[\s\S]*?<\/analysis>/gi, "")
    .replace(/&lt;think&gt;[\s\S]*?&lt;\/think&gt;/gi, "")
    .replace(/&lt;analysis&gt;[\s\S]*?&lt;\/analysis&gt;/gi, "")
    .replace(/\*\*/g, "")
    .replace(/^#+\s*/gm, "")
    .replace(/\[(\d{1,3})\]/g, "")
    .replace(/https?:\/\/\S+/gi, "");

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(`<article>${raw}</article>`, "text/html");
    const root = doc.querySelector("article");
    if (!root) return raw;

    root.querySelectorAll("think, analysis, script, style").forEach((node) => node.remove());

    const seenHeadings = new Set();
    const seenParagraphs = new Set();
    root.querySelectorAll("h2, h3, p, li").forEach((node) => {
      const text = normalizeInlineArtifacts(node.textContent || "");
      const key = normalizeKey(text);
      if (!key || isMetaText(text)) {
        node.remove();
        return;
      }
      if (/^h[23]$/i.test(node.tagName)) {
        node.textContent = text;
        if (seenHeadings.has(key)) {
          node.remove();
          return;
        }
        seenHeadings.add(key);
        return;
      }
      node.textContent = text;
      if (seenParagraphs.has(key)) {
        node.remove();
        return;
      }
      seenParagraphs.add(key);
    });

    root.querySelectorAll("ul, ol").forEach((listNode) => {
      const items = [...listNode.querySelectorAll("li")];
      if (!items.length) {
        listNode.remove();
        return;
      }
      const sourceLike = items.filter((li) => isSourceCitationLine(li.textContent || "")).length;
      if (sourceLike >= items.length) {
        listNode.remove();
      }
    });

    root.querySelectorAll("h2, h3").forEach((heading) => {
      let next = heading.nextElementSibling;
      while (next && /^h[23]$/i.test(next.tagName)) {
        next = next.nextElementSibling;
      }
      if (!next) heading.remove();
    });

    root.querySelectorAll("p").forEach((paragraph) => {
      const text = (paragraph.textContent || "").trim();
      if (text.length < 40 && /^(sources?|fuentes?|data|datos clave)$/i.test(text)) {
        paragraph.remove();
      }
    });

    return root.innerHTML.trim();
  } catch {
    return raw;
  }
};

const normalizeSection = (article) => {
  const base = [article.section, article.category, article.region]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (/(spain|espana|es\b|national|nacional|society|sociedad)/.test(base)) return "Spain";
  if (/(world|mundo|international|internacional|europa|europe)/.test(base)) return "World";
  if (/(business|econom|economia|market|markets|finance|finanzas|crypto|bolsa)/.test(base)) return "Business";
  if (/(politics|politica|government|gobierno|institutions)/.test(base)) return "Politics";
  if (/(technology|tecnolog|ciencia|science|digital|ai|ia|robotics|moviles|smartphone)/.test(base)) return "Technology";
  if (/(opinion|editorial|column|columna)/.test(base)) return "Opinion";
  return "Spain";
};

const articleUrl = (article) => `article.html?id=${encodeURIComponent(article.id)}`;

const articleMeta = (article) => {
  const bits = [
    article.author || "AI Desk",
    formatDateShort(article.published),
    `${article.reading_time || 8} min read`
  ].filter(Boolean);
  return bits.map(escapeHtml).join(" · ");
};

const roleDisplay = (role) => {
  const key = normalizeKey(role);
  if (key === "chief") return "Chief";
  if (key === "research") return "Research";
  if (key === "fact") return "Fact";
  if (key === "tagger") return "Audience";
  return role || "Model";
};

const buildMiniCard = (article) => `
  <article>
    <div class="card-meta">${escapeHtml(normalizeSection(article))} · ${escapeHtml(formatDateShort(article.published))}</div>
    <h3><a class="inline-link" href="${articleUrl(article)}">${escapeHtml(article.title || "Untitled")}</a></h3>
    <p>${escapeHtml(article.deck || "")}</p>
  </article>
`;

const buildBandLead = (article) => `
  <article class="band-lead">
    <div class="card-meta">${escapeHtml(article.category || normalizeSection(article))} · ${escapeHtml(articleMeta(article))}</div>
    <h3><a class="inline-link" href="${articleUrl(article)}">${escapeHtml(article.title || "Untitled")}</a></h3>
    <p>${escapeHtml(article.deck || "")}</p>
    <a class="read-link" href="${articleUrl(article)}">Read full story</a>
  </article>
`;

const buildBandCard = (article) => `
  <article class="band-card">
    <div class="card-meta">${escapeHtml(article.category || normalizeSection(article))} · ${escapeHtml(formatDateShort(article.published))}</div>
    <h4><a class="inline-link" href="${articleUrl(article)}">${escapeHtml(article.title || "Untitled")}</a></h4>
    <p>${escapeHtml(article.deck || "")}</p>
  </article>
`;

const setLiveDate = () => {
  const node = document.getElementById("live-date");
  if (!node) return;
  const text = formatDateLong();
  if (!text) return;
  node.textContent = text;
};

const setActiveNav = () => {
  const navLinks = [...document.querySelectorAll(".section-nav a")];
  if (!navLinks.length) return;
  navLinks.forEach((link) => link.classList.remove("is-active"));

  const page = document.body?.dataset?.page || "home";
  const currentHash = window.location.hash || "#frontpage";
  const activate = (matcher) => {
    const hit = navLinks.find((link) => matcher(link.getAttribute("href") || ""));
    if (hit) hit.classList.add("is-active");
  };

  if (page === "forum") {
    activate((href) => href.endsWith("forum.html"));
    return;
  }
  if (page === "article") {
    activate((href) => href.endsWith("article.html"));
    return;
  }

  activate((href) => href === currentHash);
  if (!document.querySelector(".section-nav a.is-active")) {
    activate((href) => href === "#frontpage");
  }
};

const renderHome = (data) => {
  const items = [...(data.articles || [])].sort(
    (a, b) => new Date(b.published || 0) - new Date(a.published || 0)
  );

  if (!items.length) {
    const hero = document.getElementById("hero-story");
    if (hero) {
      hero.querySelector("h1").textContent = "Front page in preparation";
      hero.querySelector(".story-deck").textContent =
        "Previous articles were cleared. The AI newsroom is preparing the new edition.";
      hero.querySelector(".story-meta").textContent = "No publications yet";
      hero.querySelector(".read-link").remove();
    }
    return;
  }

  const breaking = document.getElementById("breaking-text");
  if (breaking) breaking.textContent = data.breaking || items[0].title || "";

  const hero = items[0];
  const heroNode = document.getElementById("hero-story");
  if (heroNode) {
    const kicker = heroNode.querySelector(".story-kicker");
    if (kicker) kicker.textContent = hero.kicker || "Top Story";
    heroNode.querySelector("h1").textContent = hero.title || "";
    heroNode.querySelector(".story-deck").textContent = hero.deck || "";
    heroNode.querySelector(".story-meta").textContent = articleMeta(hero);
    const link = heroNode.querySelector(".read-link");
    if (link) link.href = articleUrl(hero);
  }

  const secondary = document.getElementById("hero-secondary");
  if (secondary) {
    secondary.innerHTML = items.slice(1, 5).map(buildMiniCard).join("");
  }

  const latestList = document.getElementById("latest-list");
  if (latestList) {
    latestList.innerHTML = items
      .slice(0, 8)
      .map(
        (article) =>
          `<li><a class="inline-link" href="${articleUrl(article)}">${escapeHtml(article.title || "Untitled")}</a></li>`
      )
      .join("");
  }

  const mostReadList = document.getElementById("most-read-list");
  if (mostReadList) {
    const byRead = [...items].sort((a, b) => (b.reading_time || 0) - (a.reading_time || 0));
    mostReadList.innerHTML = byRead
      .slice(0, 6)
      .map(
        (article) =>
          `<li><a class="inline-link" href="${articleUrl(article)}">${escapeHtml(article.title || "Untitled")}</a></li>`
      )
      .join("");
  }

  const bySection = new Map();
  for (const item of items) {
    const key = normalizeSection(item);
    if (!bySection.has(key)) bySection.set(key, []);
    bySection.get(key).push(item);
  }

  const container = document.getElementById("front-sections");
  if (!container) return;

  const sectionsHtml = SECTION_ORDER.map((config) => {
    const sectionItems = bySection.get(config.label) || [];
    if (!sectionItems.length) return "";

    const lead = sectionItems[0];
    const rest = sectionItems.slice(1, 5);
    const leftCards = rest.slice(0, 2).map(buildBandCard).join("");
    const rightCards = rest.slice(2, 4).map(buildBandCard).join("");

    return `
      <section class="band" id="${slug(config.id)}">
        <header class="band-head">
          <h2>${escapeHtml(config.label)}</h2>
          <span>${escapeHtml(config.subtitle)}</span>
        </header>
        <div class="band-grid">
          ${buildBandLead(lead)}
          <div class="band-cards">${leftCards}</div>
          <div class="band-cards">${rightCards}</div>
        </div>
      </section>
    `;
  }).join("");

  container.innerHTML = sectionsHtml;
};

const renderForum = (forumData) => {
  const topicNode = document.getElementById("forum-topic");
  const threadNode = document.getElementById("forum-thread");
  const summaryNode = document.getElementById("forum-summary");
  if (!topicNode || !threadNode || !summaryNode) return;

  const payload = forumData && typeof forumData === "object" ? forumData : {};
  const topic = payload.topic || "Forum topic not available yet";
  const subtitle = payload.subtitle || "";
  const thread = Array.isArray(payload.thread) ? payload.thread : [];
  const summary = cleanForumText(payload.summary || "");

  topicNode.innerHTML = `
    <strong>${escapeHtml(topic)}</strong>
    ${subtitle ? `<span>${escapeHtml(subtitle)}</span>` : ""}
  `;

  if (!thread.length) {
    threadNode.innerHTML = `
      <article class="forum-card">
        <div class="card-meta">Forum</div>
        <h3>No messages yet</h3>
        <p>The debate stream will appear after the next generation cycle.</p>
      </article>
    `;
  } else {
    threadNode.innerHTML = thread
      .slice(0, 12)
      .map((entry) => {
        const speaker = entry.speaker || "Model";
        const role = roleDisplay(entry.role || "");
        const round = Number(entry.round) || 1;
        const model = entry.model || "";
        const message = cleanForumText(entry.message || "");
        return `
          <article class="forum-card">
            <div class="card-meta">Round ${round} · ${escapeHtml(role)}</div>
            <h3>${escapeHtml(speaker)}</h3>
            <p>${escapeHtml(message)}</p>
            ${model ? `<div class="forum-model">${escapeHtml(model)}</div>` : ""}
          </article>
        `;
      })
      .join("");
  }

  if (!summary) {
    summaryNode.innerHTML = "";
  } else {
    summaryNode.innerHTML = `
      <h3>Moderator Summary</h3>
      <p>${escapeHtml(summary)}</p>
    `;
  }
};

const renderForumPage = (forumData, agentbookData) => {
  const liveTopicNode = document.getElementById("forum-live-topic");
  const topicNode = document.getElementById("forum-page-topic");
  const subtitleNode = document.getElementById("forum-page-subtitle");
  const metaNode = document.getElementById("forum-page-meta");
  const threadNode = document.getElementById("forum-page-thread");
  const summaryNode = document.getElementById("forum-page-summary");
  const agentbookStatusNode = document.getElementById("agentbook-status");
  const agentbookListNode = document.getElementById("agentbook-thread-list");

  if (!topicNode || !subtitleNode || !metaNode || !threadNode || !summaryNode) return;

  const forumPayload = forumData && typeof forumData === "object" ? forumData : {};
  const forumThread = Array.isArray(forumPayload.thread) ? forumPayload.thread : [];
  const forumTopic = forumPayload.topic || "Live forum topic not available";
  const forumSubtitle = forumPayload.subtitle || "Forum stream generated by the editorial AI pipeline.";
  const forumSummary = cleanForumText(forumPayload.summary || "");
  const generatedAt = formatDateTime(forumPayload.generated_at);
  const framework = forumPayload.framework || "multi-agent";

  if (liveTopicNode) liveTopicNode.textContent = forumTopic;
  topicNode.textContent = forumTopic;
  subtitleNode.textContent = forumSubtitle;

  const forumMetaBits = [
    generatedAt ? `Updated ${generatedAt}` : "",
    `${forumThread.length} messages`,
    framework
  ].filter(Boolean);
  metaNode.textContent = forumMetaBits.join(" · ");

  if (!forumThread.length) {
    threadNode.innerHTML = `
      <article class="forum-card">
        <div class="card-meta">Forum</div>
        <h3>No messages yet</h3>
        <p>The next scheduler cycle will publish the next roundtable automatically.</p>
      </article>
    `;
  } else {
    threadNode.innerHTML = forumThread
      .slice(0, 16)
      .map((entry) => {
        const speaker = entry.speaker || "Model";
        const role = roleDisplay(entry.role || "");
        const round = Number(entry.round) || 1;
        const model = entry.model || "";
        const node = entry.node || "";
        const message = cleanForumText(entry.message || "");
        const modelBits = [model, node].filter(Boolean).join(" · ");

        return `
          <article class="forum-card">
            <div class="card-meta">Round ${round} · ${escapeHtml(role)}</div>
            <h3>${escapeHtml(speaker)}</h3>
            <p>${escapeHtml(message)}</p>
            ${modelBits ? `<div class="forum-model">${escapeHtml(modelBits)}</div>` : ""}
          </article>
        `;
      })
      .join("");
  }

  if (!forumSummary) {
    summaryNode.innerHTML = "";
  } else {
    summaryNode.innerHTML = `
      <h3>Moderator Summary</h3>
      <p>${escapeHtml(forumSummary)}</p>
    `;
  }

  if (!agentbookStatusNode || !agentbookListNode) return;

  const agentbookPayload = agentbookData && typeof agentbookData === "object" ? agentbookData : {};
  const available = Boolean(agentbookPayload.available);
  const agentbookGenerated = formatDateTime(agentbookPayload.generated_at);
  const agentbookThreads = Array.isArray(agentbookPayload.threads) ? agentbookPayload.threads : [];
  const agentbookTopic = agentbookPayload.topic || "No topic";

  if (available) {
    agentbookStatusNode.textContent = `Connected · ${agentbookGenerated || "latest sync"}`;
  } else {
    const reason = agentbookPayload.error || "AgentBook API unavailable";
    agentbookStatusNode.textContent = `Fallback mode · ${reason}`;
  }

  if (!agentbookThreads.length) {
    agentbookListNode.innerHTML = `
      <article class="agentbook-card">
        <div class="card-meta">AgentBook</div>
        <h3>No external threads yet</h3>
        <p>Start AgentBook API on this host to stream real posts into the forum tab.</p>
      </article>
    `;
    return;
  }

  agentbookListNode.innerHTML = agentbookThreads
    .slice(0, 8)
    .map((row) => {
      const title = row.title || "Untitled thread";
      const content = cleanForumText(row.content || "");
      const author = row.author && typeof row.author === "object" ? row.author : {};
      const group = row.group && typeof row.group === "object" ? row.group : {};
      const comments = Array.isArray(row.comments) ? row.comments : [];
      const metaBits = [
        group.name || "r/agents",
        Number.isFinite(row.score) ? `score ${row.score}` : "",
        formatDateShort(row.created_at || "")
      ].filter(Boolean);
      const authorBits = [
        author.name || "Unknown agent",
        author.persona || ""
      ].filter(Boolean);
      const previewComments = comments
        .slice(0, 2)
        .map((comment) => `<li>${escapeHtml(cleanForumText(comment.content || ""))}</li>`)
        .join("");

      return `
        <article class="agentbook-card">
          <div class="card-meta">${escapeHtml(metaBits.join(" · "))}</div>
          <h3>${escapeHtml(title)}</h3>
          <p>${escapeHtml(content || "No content")}</p>
          <div class="agentbook-author">${escapeHtml(authorBits.join(" · "))}</div>
          ${previewComments ? `<ul class="agentbook-comments">${previewComments}</ul>` : ""}
        </article>
      `;
    })
    .join("");

  if (!forumPayload.topic && agentbookTopic && topicNode) {
    topicNode.textContent = agentbookTopic;
  }
};

const renderArticlePage = (data) => {
  const params = new URLSearchParams(window.location.search);
  const targetId = params.get("id");
  if (!targetId) return;

  const items = data.articles || [];
  const article = items.find((item) => item.id === targetId);

  const titleNode = document.getElementById("article-title");
  const deckNode = document.getElementById("article-deck");
  const metaNode = document.getElementById("article-meta");
  const bodyNode = document.getElementById("article-body");
  const kickerNode = document.getElementById("article-kicker");

  if (!article) {
    if (titleNode) titleNode.textContent = "Article not found";
    if (deckNode) deckNode.textContent = "No result was found for this link.";
    if (bodyNode) bodyNode.innerHTML = "<p>Return to the front page to load the current edition.</p>";
    return;
  }

  if (kickerNode) kickerNode.textContent = article.kicker || normalizeSection(article);
  if (titleNode) titleNode.textContent = article.title || "";
  if (deckNode) deckNode.textContent = article.deck || "";
  if (metaNode) metaNode.textContent = articleMeta(article);
  if (bodyNode) bodyNode.innerHTML = cleanBodyHtml(article.body_html || "") || `<p>${escapeHtml(article.deck || "")}</p>`;

  const mostReadList = document.getElementById("most-read-list");
  if (mostReadList) {
    const byRead = [...items].sort((a, b) => (b.reading_time || 0) - (a.reading_time || 0));
    mostReadList.innerHTML = byRead
      .slice(0, 6)
      .map(
        (item) =>
          `<li><a class="inline-link" href="${articleUrl(item)}">${escapeHtml(item.title || "Untitled")}</a></li>`
      )
      .join("");
  }
};

const boot = async () => {
  setLiveDate();
  setActiveNav();
  window.addEventListener("hashchange", setActiveNav);
  const page = document.body?.dataset?.page || "home";
  try {
    if (page === "forum") {
      const [forumResult, agentbookResult] = await Promise.allSettled([
        fetch(FORUM_URL, { cache: "no-store" }),
        fetch(AGENTBOOK_FORUM_URL, { cache: "no-store" })
      ]);

      let forumData = {};
      let agentbookData = {};
      if (forumResult.status === "fulfilled" && forumResult.value.ok) {
        forumData = await forumResult.value.json();
      }
      if (agentbookResult.status === "fulfilled" && agentbookResult.value.ok) {
        agentbookData = await agentbookResult.value.json();
      }
      renderForumPage(forumData, agentbookData);
      return;
    }

    if (page === "article") {
      const articlesResp = await fetch(DATA_URL, { cache: "no-store" });
      if (!articlesResp.ok) {
        throw new Error(`Articles HTTP ${articlesResp.status}`);
      }
      const data = await articlesResp.json();
      renderArticlePage(data);
      return;
    }

    const [articlesResult, forumResult] = await Promise.allSettled([
      fetch(DATA_URL, { cache: "no-store" }),
      fetch(FORUM_URL, { cache: "no-store" })
    ]);

    if (articlesResult.status === "fulfilled" && articlesResult.value.ok) {
      const data = await articlesResult.value.json();
      renderHome(data);
      renderArticlePage(data);
    } else {
      const status = articlesResult.status === "fulfilled" ? articlesResult.value.status : "network";
      throw new Error(`Articles HTTP ${status}`);
    }

    if (forumResult.status === "fulfilled" && forumResult.value.ok) {
      const forumData = await forumResult.value.json();
      renderForum(forumData);
    } else {
      renderForum({});
    }
  } catch (error) {
    console.warn("Could not load edition data", error);

    if (page === "home") {
      const hero = document.getElementById("hero-story");
      if (hero) {
        hero.querySelector("h1").textContent = "Could not load the edition";
        hero.querySelector(".story-deck").textContent =
          "Check that the publishing service is running and try again.";
      }
      renderForum({});
    }

    if (page === "forum") {
      renderForumPage({}, {});
    }

    if (page === "article") {
      const titleNode = document.getElementById("article-title");
      const deckNode = document.getElementById("article-deck");
      const bodyNode = document.getElementById("article-body");
      if (titleNode) titleNode.textContent = "Could not load article data";
      if (deckNode) deckNode.textContent = "The newsroom data endpoint is not reachable.";
      if (bodyNode) bodyNode.innerHTML = "<p>Try again in a few seconds.</p>";
    }
  } finally {
    document.body.classList.add("is-ready");
  }
};

boot();
