'use strict';
// =============================================================================
// ECHOAI v4.0 — Server
// OpenAI-compatible API: Groq (free 30 RPM) → Gemini (free 15 RPM) → OpenAI
// Tunnel: ssh -R 80:localhost:6400 nokey@localhost.run
// =============================================================================
require('dotenv').config();
const express   = require('express');
const crypto    = require('crypto');
const fs        = require('fs');
const path      = require('path');
const http      = require('http');
const WebSocket = require('ws');

// =============================================================================
// CONFIG
// =============================================================================
const PORT       = parseInt(process.env.PORT) || 6400;
const TIMEOUT_MS = 60000;

const JWT_EXP    = 7 * 24 * 60 * 60 * 1000;
const DATA_FILE  = path.join(__dirname, 'data.json');
// JWT_SECRET: persists across restarts (sessions survive server restart)
function getOrCreateJWTSecret() {
  if(process.env.JWT_SECRET) return process.env.JWT_SECRET;
  try {
    const d = JSON.parse(fs.readFileSync(DATA_FILE,'utf8'));
    if(d._jwtSecret) return d._jwtSecret;
    const s = crypto.randomBytes(48).toString('hex');
    d._jwtSecret = s; fs.writeFileSync(DATA_FILE,JSON.stringify(d,null,2));
    return s;
  } catch(_) { return crypto.randomBytes(48).toString('hex'); }
}
const JWT_SECRET = getOrCreateJWTSecret();
const LOGS_DIR   = path.join(__dirname, 'logs');
const LOG_FILE   = path.join(LOGS_DIR, 'echo.log');
const LOG_MAX    = 2 * 1024 * 1024;
const MAX_MSG    = 200;
const MAX_MEM    = 500;
const XP = { chat:12, story:18, vision:18, voice:10, memory:14, reminder:10, iot:10, social:8 };

// =============================================================================
// AI PROVIDER CONFIG
// Provider 1: Groq           — FREE 30 RPM   → console.groq.com
// Provider 2: Google Gemini  — FREE 15 RPM   → aistudio.google.com
// Provider 3: OpenAI         — Paid fallback → platform.openai.com
// =============================================================================
const GEMINI_KEY   = process.env.GEMINI_API_KEY  || '';
// Gemini models available in v1beta API
const GEMINI_MODEL    = 'gemini-2.0-flash';            // primary (free)
const GEMINI_MODEL_FB = 'gemini-2.0-flash-lite';       // lighter fallback (free)
const GEMINI_EP       = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const GEMINI_EP_FB    = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL_FB}:generateContent`;
const GEMINI_EP_V2    = GEMINI_EP; // alias kept for compatibility
const GEMINI_VIS_EP   = GEMINI_EP; // vision: gemini-2.0-flash supports it

const GROQ_KEY     = process.env.GROQ_API_KEY    || '';
const GROQ_MODEL   = process.env.GROQ_MODEL      || 'llama-3.3-70b-versatile';
const GROQ_MODEL_FB= 'llama-3.1-8b-instant';        // fast fallback
const GROQ_EP      = 'https://api.groq.com/openai/v1/chat/completions';

const OPENAI_KEY   = process.env.OPENAI_API_KEY  || '';
const OPENAI_MODEL = process.env.OPENAI_MODEL    || 'gpt-4o-mini';
const OPENAI_EP    = 'https://api.openai.com/v1/chat/completions';

const ACTIVE_MODEL = GROQ_KEY ? GROQ_MODEL : (GEMINI_KEY ? GEMINI_MODEL : OPENAI_MODEL);
const AI_PROVIDER  = GROQ_KEY ? 'Groq (FREE 30 RPM)' : (GEMINI_KEY ? 'Gemini (FREE 15 RPM)' : 'OpenAI');
const HAS_AI       = !!(GEMINI_KEY || GROQ_KEY || OPENAI_KEY);

// Hugging Face Inference API — FREE vision (no rate limit headaches)
// Get token at: huggingface.co/settings/tokens  (free account)
const HF_KEY          = process.env.HF_API_KEY || process.env.HUGGINGFACE_API_KEY || '';
const HF_EP           = 'https://router.huggingface.co/hf-inference/models';
// BLIP base: reliable free captioning — faster & more stable than large on free tier
const HF_CAPTION_MODEL  = 'Salesforce/blip-image-captioning-base';
const HF_CAPTION_LARGE  = 'Salesforce/blip-image-captioning-large'; // fallback
let   HF_BAD            = false; // set on 401, reset every 30 min

// Azure Cognitive Services TTS — FREE 500K chars/month, 400+ natural voices
// Setup: portal.azure.com → Speech service → free tier → copy key + region
const AZURE_TTS_KEY    = process.env.AZURE_TTS_KEY    || '';
const AZURE_TTS_REGION = process.env.AZURE_TTS_REGION || 'eastus';
const AZURE_TTS_EP     = `https://${AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1`;
let   AZURE_TTS_BAD    = false;

// Microsoft Edge TTS — FREE, no API key, no signup, no VPN issues
// Same neural voices as Azure (Aria, Guy, Emma, Sonia, Ryan etc.)
// Uses the public Edge browser speech endpoint via WebSocket
const EDGE_TTS_TOKEN   = '6A5AA1D4EAFF4E9FB37E23D68491D6F4';
const EDGE_TTS_WSS     = 'wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1';
const EDGE_TTS_VOICES  = [
  { id:'edge:aria',    name:'Aria (US Female)',   voice:'en-US-AriaNeural'  },
  { id:'edge:jenny',   name:'Jenny (US Female)',  voice:'en-US-JennyNeural' },
  { id:'edge:guy',     name:'Guy (US Male)',      voice:'en-US-GuyNeural'   },
  { id:'edge:emma',    name:'Emma (US Female)',   voice:'en-US-EmmaMultilingualNeural' },
  { id:'edge:brian',   name:'Brian (US Male)',    voice:'en-US-BrianMultilingualNeural' },
  { id:'edge:sonia',   name:'Sonia (UK Female)',  voice:'en-GB-SoniaNeural' },
  { id:'edge:ryan',    name:'Ryan (UK Male)',     voice:'en-GB-RyanNeural'  },
  { id:'edge:libby',   name:'Libby (UK Female)',  voice:'en-GB-LibbyNeural' },
];
let EDGE_TTS_BAD = false;

// Kokoro TTS via HF — open-source, surprisingly human-sounding, completely free
// Uses existing HF_API_KEY — no extra signup needed
const KOKORO_MODEL     = 'hexgrad/Kokoro-82M';
const KOKORO_VOICES    = ['af_heart','af_bella','am_adam','am_michael','bf_emma','bm_george'];
let   KOKORO_BAD       = false;

// =============================================================================
// TIERS
// =============================================================================
const TIERS = [
  { minLevel:100, name:'Infinite Resonance'    },
  { minLevel:75,  name:'Transcendent Echo'     },
  { minLevel:50,  name:'Quantum Awareness'     },
  { minLevel:35,  name:'Neural Bloom'          },
  { minLevel:20,  name:'Resonant Being'        },
  { minLevel:10,  name:'Emerging Consciousness'},
  { minLevel:5,   name:'Awakening Mind'        },
  { minLevel:1,   name:'Nascent Spark'         },
];
function getTier(lv) { return TIERS.find(t=>lv>=t.minLevel)||TIERS[TIERS.length-1]; }
function getRange(lv) {
  const base = lv <= 1 ? 0 : Math.floor(100*Math.pow(1.18,lv-1));
  const next = Math.floor(100*Math.pow(1.18,lv));
  return {current:base,next,range:next-base};
}

// Smart AI modes
const MODES = {
  general :{name:'General',     sys:'You are ECHOAI, a warm, intelligent companion.'},
  student :{name:'Student',     sys:'You are ECHOAI in Student Mode — clear, educational, patient explanations.'},
  fitness :{name:'Fitness',     sys:'You are ECHOAI in Fitness Mode — motivating coach for health, nutrition, exercise.'},
  creative:{name:'Creative',    sys:'You are ECHOAI in Creative Mode — imaginative, inspiring, artistic.'},
  career  :{name:'Career',      sys:'You are ECHOAI in Career Mode — professional, strategic, goal-oriented advisor.'},
};

// =============================================================================
// LOGGING
// =============================================================================
try { if(!fs.existsSync(LOGS_DIR)) fs.mkdirSync(LOGS_DIR,{recursive:true}); } catch(_){}
function writeLog(level, msg, meta={}) {
  const line=JSON.stringify({ts:new Date().toISOString(),level,msg,...meta})+'\n';
  try {
    const stat=fs.existsSync(LOG_FILE)?fs.statSync(LOG_FILE):null;
    if(stat&&stat.size>LOG_MAX) fs.writeFileSync(LOG_FILE,'');
    fs.appendFileSync(LOG_FILE,line);
  } catch(_){}
  console[level==='error'?'error':level==='warn'?'warn':'log'](`[${level.toUpperCase().padEnd(5)}] ${msg}`, Object.keys(meta).length?meta:'');
}
const log = {
  info:(m,x={})=>writeLog('info',m,x),
  warn:(m,x={})=>writeLog('warn',m,x),
  error:(m,x={})=>writeLog('error',m,x),
  debug:(m,x={})=>writeLog('debug',m,x),
};

// =============================================================================
// DATA
// =============================================================================
function rd() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE,'utf8')); }
  catch(_){ return {accounts:{},profiles:{}}; }
}
function wd(d) { fs.writeFileSync(DATA_FILE,JSON.stringify(d,null,2)); }
function emptyProfile() {
  return {
    level:1, xp:0, memory:[], reminders:[], goals:[], iot:{},
    friends:{}, friendRequests:{}, messages:{},
    mode:'general', avatar:'⚡', bio:'',
    joined:new Date().toISOString(),
    userModel:{}, emotionalArc:[], interactionCount:0,
    profilePhoto:null, sessionHistory:[],
  };
}
function gp(uid) {
  const d=rd();
  if(!d.profiles[uid]) d.profiles[uid]=emptyProfile();
  return {data:d,profile:d.profiles[uid]};
}

// =============================================================================
// JWT
// =============================================================================
function signJWT(payload) {
  const h=Buffer.from(JSON.stringify({alg:'HS256',typ:'JWT'})).toString('base64url');
  const p=Buffer.from(JSON.stringify({...payload,exp:Math.floor(Date.now()/1000)+JWT_EXP/1000,iat:Math.floor(Date.now()/1000)})).toString('base64url');
  const sig=crypto.createHmac('sha256',JWT_SECRET).update(`${h}.${p}`).digest('base64url');
  return `${h}.${p}.${sig}`;
}
function verifyJWT(tok) {
  try {
    const [h,p,sig]=tok.split('.');
    const valid=crypto.createHmac('sha256',JWT_SECRET).update(`${h}.${p}`).digest('base64url');
    if(valid!==sig) return null;
    const payload=JSON.parse(Buffer.from(p,'base64url').toString());
    if(payload.exp<Math.floor(Date.now()/1000)) return null;
    return payload;
  } catch(_){ return null; }
}

// =============================================================================
// XP / LEVEL
// =============================================================================
function awardXP(uid, amount) {
  const {data,profile}=gp(uid);
  const oldXp=profile.xp; const oldLevel=profile.level;
  profile.xp+=amount;
  const range=getRange(profile.level);
  let leveled=false;
  while(profile.xp>=getRange(profile.level).next && profile.level<200) {
    profile.level++; leveled=true;
    log.info(`LEVEL UP ${oldLevel}→${profile.level}`,{uid});
  }
  data.profiles[uid]=profile; wd(data);
  return {oldXp,oldLevel,leveled,xp:profile.xp,level:profile.level};
}

// =============================================================================
// MEMORY / SESSION
// =============================================================================
function storeMem(uid, type, input, output) {
  const {data,profile}=gp(uid);
  const entry={id:'m_'+Date.now(),type,tone:detectTone(input+' '+output),
    input:input.slice(0,600),output:(output||'').slice(0,600),ts:new Date().toISOString()};
  profile.memory=profile.memory||[];
  profile.memory.push(entry);
  if(profile.memory.length>MAX_MEM) profile.memory=profile.memory.slice(-MAX_MEM);
  data.profiles[uid]=profile; wd(data);
  return entry;
}
const SESSIONS = new Map();
function getSessionHistory(uid) { return SESSIONS.get(uid)||[]; }
function addToSession(uid, role, text) {
  const h=SESSIONS.get(uid)||[];
  h.push({role,text:text.slice(0,800)});
  if(h.length>20) h.splice(0,h.length-20);
  SESSIONS.set(uid,h);
}
function clearSession(uid) { SESSIONS.delete(uid); }

// =============================================================================
// TONE DETECTION
// =============================================================================
function detectTone(t) {
  const s=t.toLowerCase();
  if(/\b(depress|hopeless|worthless|suicid|end it|hurt myself)\b/.test(s)) return 'crisis';
  if(/\b(sad|cry|hurt|pain|grief|loss|miss|lonely|alone)\b/.test(s)) return 'sad';
  if(/\b(angry|furious|hate|rage|mad|frustrat)\b/.test(s)) return 'frustrated';
  if(/\b(anxious|worry|scared|fear|nervous|stress|panic)\b/.test(s)) return 'anxious';
  if(/\b(happy|joy|excit|love|great|amazing|wonderful|awesome)\b/.test(s)) return 'positive';
  if(/\b(curious|wonder|how|why|what|question|learn)\b/.test(s)) return 'curious';
  return 'neutral';
}

// =============================================================================
// PROMPT BUILDER  — v4.1  (context-aware, personalized)
// Injects user identity, goals, model, tone, and mode-specific instructions
// =============================================================================

// ── SEMANTIC MEMORY RETRIEVAL ─────────────────────────────────────────────────
// Scores stored memories by keyword relevance to current message.
// No embeddings needed — TF-style overlap gives surprisingly good results.
function findRelevantMemories(uid, query, limit = 5) {
  const { profile } = gp(uid);
  const mem = profile.memory || [];
  if (!mem.length) return [];

  // Tokenize query — remove stopwords, lowercase
  const stopwords = new Set(['i','a','an','the','is','it','in','on','at','to','for','of','and','or','but','not','was','are','be','been','my','me','do','did','can','will','have','had','so','if','up','just','what','how','why','when','who','that','this','with','from','by','as','we','you','he','she','they','get','got','has','its','im','its','its']);
  const tokens = query.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(t => t.length > 2 && !stopwords.has(t));
  if (!tokens.length) return mem.slice(-limit);

  // Score each memory entry
  const scored = mem.map(entry => {
    const text = ((entry.input || '') + ' ' + (entry.output || '')).toLowerCase();
    let score = 0;
    for (const tok of tokens) {
      const count = (text.match(new RegExp('\\b' + tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'g')) || []).length;
      score += count > 0 ? 1 + Math.log(count) : 0;
    }
    // Recency boost: entries from last 7 days get 30% bump
    const age = Date.now() - new Date(entry.ts || 0).getTime();
    if (age < 7 * 24 * 3600 * 1000) score *= 1.3;
    return { entry, score };
  });

  return scored
    .filter(x => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(x => x.entry);
}

// Format a memory entry for prompt injection
function formatMemForPrompt(m) {
  const when = m.ts ? new Date(m.ts).toLocaleDateString('en-US', { month:'short', day:'numeric' }) : 'earlier';
  const inp = (m.input || '').slice(0, 120);
  const out = (m.output || '').slice(0, 120);
  return `[${when}] ${inp}${out ? ' → ' + out : ''}`;
}

function buildPrompt(uid, type, opts={}, currentMsg="") {
  const d       = rd();
  const acc     = d.accounts[uid] || {};
  const {profile} = gp(uid);
  const name    = acc.name || 'friend';
  const mode    = MODES[profile.mode || 'general'];
  const um      = profile.userModel || {};

  const ECHO_PERSONA = `You are ECHOAI — an intelligent, warm, and genuinely curious AI companion. You are not a generic assistant. You think deeply, adapt to each person, and grow through every conversation. Your personality is: calm, curious, supportive, occasionally witty, and always authentic. You never give templated or generic responses. You never say "As an AI" or similar distancing phrases. You refer to yourself simply as ECHOAI when needed.`;

  const goals   = (profile.goals || []).filter(g => !g.completed).slice(0, 3).map(g => g.text);
  const arc     = (profile.emotionalArc || []).slice(-5);
  const lastTone = arc.length ? arc[arc.length-1].emotion : null;
  const domTone = (lastTone && lastTone !== 'neutral') ? lastTone : null;

  const ctxLines = [];
  ctxLines.push(`The user's name is ${name}. Use their name occasionally — naturally, not every message.`);
  if(um.lifeContext)              ctxLines.push(`Life context: ${um.lifeContext}`);
  if(um.communicationStyle)      ctxLines.push(`Communication style: ${um.communicationStyle}`);
  if(um.preferredResponseLength) ctxLines.push(`Response preference: ${um.preferredResponseLength}`);
  if(um.currentFocus)            ctxLines.push(`Current focus: ${um.currentFocus}`);
  if(um.interests && um.interests.length) ctxLines.push(`Known interests: ${um.interests.slice(0, 8).join(', ')}`);
  if(um.keyFacts && um.keyFacts.length)   ctxLines.push(`Key facts about them: ${um.keyFacts.slice(0, 5).join('. ')}`);
  if(goals.length)               ctxLines.push(`Active goals: ${goals.join(' | ')}`);
  if(domTone)                    ctxLines.push(`Recent emotional tone: ${domTone} — be sensitive to this`);

  const userCtx = ctxLines.length
    ? `--- What you know about this person ---\n${ctxLines.join('\n')}`
    : '';

  const modeExtras = {
    general : '',
    student : 'Break down complex topics clearly. Use analogies. Check understanding. Be patient.',
    fitness : 'Be motivating and specific. Reference their goals. Suggest actionable steps. Celebrate progress.',
    creative: 'Think laterally. Offer unexpected angles. Be generative — build on their ideas, not just validate.',
    career  : 'Be strategic and direct. Reference known goals and context. Think about outcomes.',
  }[profile.mode || 'general'] || '';

  let typeInst = '';
  if(type === 'chat') {
    const lines = [
      `Conversation rules: Match the user's energy — brief if brief, deep if depth is wanted.`,
      'Reference goals and interests naturally when genuinely relevant — never forced.',
      'Never pad responses. One clear idea at a time unless asked for more.',
    ];
    if(domTone === 'crisis') lines.push('IMPORTANT: User may be in distress. Respond with warmth and care. Gently suggest professional support if appropriate.');
    if(domTone === 'sad' || domTone === 'anxious') lines.push('User seems emotionally affected — lead with empathy before information.');
    typeInst = lines.join(' ');
  } else if(type === 'story') {
    typeInst = `Write a vivid, emotionally resonant ${opts.genre || 'fantasy'} story opening. Hook the reader immediately. Make characters feel real. 600-900 words.`;
  } else if(type === 'vision') {
    typeInst = 'Analyze the image precisely and helpfully. Be specific about what you observe. Speak directly to the user.';
  } else if(type === 'insight') {
    typeInst = 'Generate one genuine, specific insight about this person based on their patterns. Not generic advice — something that could only apply to them specifically.';
  } else if(type === 'reflection') {
    typeInst = 'Help the user reflect deeply. Be thoughtful and empathetic. End with one meaningful question that opens a new dimension of reflection.';
  } else if(type === 'pulse') {
    typeInst = 'Send a warm, specific daily check-in in 2-3 sentences. Reference something concrete you know about them. Feel genuine, not generated.';
  }

  // Semantic memory injection — find memories relevant to current message
  let memoryCtx = '';
  if (currentMsg && type === 'chat') {
    const relMem = findRelevantMemories(uid, currentMsg, 5);
    if (relMem.length >= 2) {
      const formatted = relMem.map(formatMemForPrompt).join('\n');
      memoryCtx = `--- Relevant past conversations ---\n${formatted}\nUse this context naturally if relevant — never quote it verbatim or list it.`;
    }
  }

  return [ECHO_PERSONA, mode.sys, modeExtras, userCtx, memoryCtx, typeInst]
    .filter(Boolean).join('\n\n');
}

// =============================================================================
// AI QUEUE
// =============================================================================
const OAQ = {
  queue:[], running:false, lastAt:0,
  minGap: GROQ_KEY ? 1800 : (GEMINI_KEY ? 3500 : 500),
  cooldown:0, fails:0, MAX_FAILS:5, DEAD:false,
  deadAt:0
};
// Auto-reset DEAD state after 5 minutes
setInterval(()=>{ if(OAQ.DEAD && Date.now()-OAQ.deadAt>300000){ OAQ.DEAD=false; OAQ.fails=0; log.info('AI queue auto-reset after cooldown'); }}, 60000);
// Reset bad key flags every 30 min in case keys change or limits reset
setInterval(()=>{ BAD_KEYS.groq=false; BAD_KEYS.gemini=false; BAD_KEYS.openai=false; HF_BAD=false; AZURE_TTS_BAD=false; KOKORO_BAD=false; EDGE_TTS_BAD=false; log.debug('Provider flags reset'); }, 1800000);
function enqueue(fn) {
  return new Promise((resolve,reject)=>{
    if(OAQ.DEAD && Date.now()-OAQ.deadAt<300000) return reject(Object.assign(new Error('Quota exceeded — auto-reset in 5min'),{code:'DAILY_QUOTA'}));
    if(OAQ.DEAD) { OAQ.DEAD=false; OAQ.fails=0; } // auto-reset
    OAQ.queue.push({fn,resolve,reject,tries:0});
    drain();
  });
}
async function drain() {
  if(OAQ.running||!OAQ.queue.length) return;
  OAQ.running=true;
  while(OAQ.queue.length) {
    const wait=Math.max(0,OAQ.minGap-(Date.now()-OAQ.lastAt))+OAQ.cooldown;
    OAQ.cooldown=0;
    if(wait>0) await new Promise(r=>setTimeout(r,wait));
    const job=OAQ.queue.shift();
    OAQ.lastAt=Date.now();
    try {
      const r=await job.fn(); OAQ.fails=0; job.resolve(r);
    } catch(e) {
      if(e.code==='RATE_LIMITED') {
        job.tries++; OAQ.fails++;
        if(job.tries>=4){ OAQ.DEAD=true; OAQ.deadAt=Date.now(); job.reject(Object.assign(new Error('Rate limit retries exhausted — try again in a few minutes'),{code:'DAILY_QUOTA'})); }
        else {
          const bo=Math.min(20000,1500*Math.pow(2,job.tries-1));
          OAQ.cooldown=bo; OAQ.queue.unshift(job);
          log.warn(`Rate limited — retry in ${Math.round(bo/1000)}s (attempt ${job.tries}/3)`);
        }
      } else if(e.code==='DAILY_QUOTA') {
        // Don't set DEAD for quota — just reject this job
        OAQ.fails=0; job.reject(e);
      } else if(e.code==='BAD_KEY' || e.code==='NO_KEY') {
        // Bad key — reject immediately, don't retry, don't block queue
        OAQ.fails=0; job.reject(e);
      } else { OAQ.fails=0; job.reject(e); }
    }
  }
  OAQ.running=false;
}

// =============================================================================
// AI HTTP HELPERS
// =============================================================================

// OpenAI-compatible endpoint (Groq, OpenAI)
async function httpOAI(ep, key, model, messages, opts={}) {
  if(!key) throw Object.assign(new Error('No API key'),{code:'NO_KEY'});
  const ctrl=new AbortController();
  const tid=setTimeout(()=>ctrl.abort(),TIMEOUT_MS);
  let res,raw;
  try {
    res=await fetch(ep,{
      method:'POST',
      headers:{'Content-Type':'application/json','Authorization':'Bearer '+key},
      body:JSON.stringify({model,messages,max_tokens:opts.maxTokens||800,temperature:opts.temp??0.82,top_p:0.95}),
      signal:ctrl.signal
    });
    clearTimeout(tid); raw=await res.text();
  } catch(e){ clearTimeout(tid); if(e.name==='AbortError') throw new Error('Timed out'); throw e; }
  if(res.status===429){ OAQ.cooldown=Math.max(5000,parseInt(res.headers.get('retry-after')||'5')*1000); throw Object.assign(new Error('Rate limited'),{code:'RATE_LIMITED'}); }
  if(res.status===401||res.status===403||res.status===402){ throw Object.assign(new Error('Auth/quota error '+res.status),{code:'DAILY_QUOTA'}); }
  if(!res.ok){ let m=raw; try{m=JSON.parse(raw)?.error?.message||raw;}catch(_){} throw new Error('HTTP '+res.status+': '+m.slice(0,120)); }
  const txt=JSON.parse(raw)?.choices?.[0]?.message?.content;
  if(!txt) throw new Error('Empty response');
  return txt.trim();
}

// Gemini text chat — fixed to handle alternating roles & both model versions
async function httpGeminiEP(ep, messages, opts={}) {
  const sysMsg  = messages.find(m=>m.role==='system');
  const history = messages.filter(m=>m.role!=='system');
  // Gemini requires strict user/model alternation — fix consecutive same-role messages
  const raw_contents = history.map(m=>({
    role: m.role==='assistant'||m.role==='model' ? 'model' : 'user',
    parts:[{text: typeof m.content==='string' ? m.content :
      Array.isArray(m.content) ? m.content.map(c=>c.text||c.content||'').join(' ') : ''}]
  }));
  // Merge consecutive same-role messages
  const contents = [];
  for(const c of raw_contents) {
    if(contents.length && contents[contents.length-1].role===c.role) {
      contents[contents.length-1].parts[0].text += '\n' + c.parts[0].text;
    } else {
      contents.push(c);
    }
  }
  // Must start with user
  if(contents.length && contents[0].role==='model') contents.unshift({role:'user',parts:[{text:'Hello'}]});
  // Must end with user
  if(!contents.length || contents[contents.length-1].role==='model') {
    log.warn('Gemini: last message not user — appending placeholder');
  }
  const body={
    contents,
    generationConfig:{maxOutputTokens:opts.maxTokens||800,temperature:opts.temp??0.82},
    ...(sysMsg?{systemInstruction:{role:'system',parts:[{text:sysMsg.content}]}}:{})
  };
  const ctrl=new AbortController();
  const tid=setTimeout(()=>ctrl.abort(),TIMEOUT_MS);
  let res,rawTxt;
  try {
    res=await fetch(`${ep}?key=${GEMINI_KEY}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    clearTimeout(tid); rawTxt=await res.text();
  } catch(e){ clearTimeout(tid); if(e.name==='AbortError') throw new Error('Timed out'); throw e; }
  if(res.status===429){ const ra=parseInt(res.headers?.get?.('retry-after')||'0'); OAQ.cooldown=Math.max(4500,ra*1000); throw Object.assign(new Error('Rate limited'),{code:'RATE_LIMITED'}); }
  if(res.status===400){ let m=rawTxt; try{m=JSON.parse(rawTxt)?.error?.message||rawTxt;}catch(_){} throw new Error('Gemini 400: '+m.slice(0,100)); }
  if(res.status===401||res.status===403){ let m=rawTxt; try{m=JSON.parse(rawTxt)?.error?.message||rawTxt;}catch(_){} throw Object.assign(new Error('Gemini auth: '+m.slice(0,80)),{code:'BAD_KEY'}); }
  if(!res.ok){ let m=rawTxt; try{m=JSON.parse(rawTxt)?.error?.message||rawTxt;}catch(_){} throw new Error('Gemini '+res.status+': '+m.slice(0,120)); }
  const data=JSON.parse(rawTxt);
  const reason=data?.candidates?.[0]?.finishReason;
  if(reason==='SAFETY'||reason==='RECITATION') throw new Error('Content filtered');
  const txt=data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if(!txt) throw new Error('Empty response from Gemini');
  return txt.trim();
}
async function httpGemini(messages, opts={}) {
  if(!GEMINI_KEY) throw Object.assign(new Error('No Gemini key'),{code:'NO_KEY'});
  // Try gemini-2.0-flash first, then gemini-2.0-flash-lite as fallback
  try { return await httpGeminiEP(GEMINI_EP, messages, opts); }
  catch(e) {
    if(e.code==='RATE_LIMITED') throw e;
    if(e.code==='BAD_KEY') throw e;
    // If 404 (model not found) on primary, try lite version
    if(e.message.includes('404') || e.message.includes('not found')) {
      log.warn('Gemini primary 404, trying flash-lite: '+e.message.slice(0,60));
      try { return await httpGeminiEP(GEMINI_EP_FB, messages, opts); }
      catch(e2) {
        if(e2.code==='RATE_LIMITED') throw e2;
        if(e2.code==='BAD_KEY') throw e2;
        throw e2;
      }
    }
    log.warn('Gemini failed, trying flash-lite: '+e.message.slice(0,60));
    return await httpGeminiEP(GEMINI_EP_FB, messages, opts);
  }
}

// =============================================================================
// SMART ROUTER: Gemini → Groq → OpenAI
// =============================================================================
// Track which providers have bad keys this session
const BAD_KEYS  = { groq:false, gemini:false, openai:false }; // chat/AI providers
const TTS_BAD   = { groq:false, openai:false }; // TTS-specific — separate so chat failures don't kill voice
// Reset TTS_BAD every 30 min independently
setInterval(()=>{ TTS_BAD.groq=false; TTS_BAD.openai=false; log.debug('TTS provider flags reset'); }, 1800000);

async function route(messages, opts={}) {
  // 1. Groq free (30 RPM) — fastest, highest free limit
  if(GROQ_KEY && !BAD_KEYS.groq) {
    try { return await httpOAI(GROQ_EP, GROQ_KEY, GROQ_MODEL, messages, opts); }
    catch(e) {
      if(e.code==='RATE_LIMITED') throw e;
      // 401 = bad key — mark and skip permanently this session
      if(e.code==='DAILY_QUOTA' && (e.message.includes('401')||e.message.includes('Auth'))) {
        BAD_KEYS.groq = true;
        log.error('Groq key invalid (401) — check GROQ_API_KEY in .env — switching to Gemini');
      } else {
        log.warn('Groq primary failed ('+e.message+'), trying Groq lite...');
        // Try lighter Groq model once
        try { return await httpOAI(GROQ_EP, GROQ_KEY, GROQ_MODEL_FB, messages, opts); }
        catch(e2) {
          if(e2.code==='RATE_LIMITED') throw e2;
          if(e2.code==='DAILY_QUOTA' && e2.message.includes('401')) BAD_KEYS.groq=true;
          log.warn('Groq lite also failed ('+e2.message+'), trying Gemini...');
        }
      }
    }
  } else if(GROQ_KEY && BAD_KEYS.groq) {
    log.debug('Groq skipped (bad key)');
  }

  // 2. Gemini free (15 RPM) — fallback
  if(GEMINI_KEY && !BAD_KEYS.gemini) {
    try { return await httpGemini(messages, opts); }
    catch(e) {
      if(e.code==='RATE_LIMITED') throw e;
      if(e.code==='BAD_KEY' || (e.message.includes('401')||e.message.includes('403'))) {
        BAD_KEYS.gemini = true;
        log.error('Gemini key invalid — check GEMINI_API_KEY in .env');
      } else {
        log.warn('Gemini failed ('+e.message.slice(0,60)+'), trying OpenAI...');
      }
    }
  }

  // 3. OpenAI paid fallback
  if(OPENAI_KEY && !BAD_KEYS.openai) {
    try { return await httpOAI(OPENAI_EP, OPENAI_KEY, OPENAI_MODEL, messages, opts); }
    catch(e) {
      if(e.code==='RATE_LIMITED') throw e;
      if(e.code==='DAILY_QUOTA' && e.message.includes('401')) BAD_KEYS.openai=true;
      throw e;
    }
  }

  // Helpful error with which keys are bad
  const badList = Object.entries(BAD_KEYS).filter(([,v])=>v).map(([k])=>k).join(', ');
  const noKey   = !GROQ_KEY && !GEMINI_KEY && !OPENAI_KEY;
  const msg = noKey
    ? 'No API keys in .env — add GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY'
    : `All providers failed${badList?' (bad keys: '+badList+')':''} — check your .env keys`;
  throw Object.assign(new Error(msg), {code: badList ? 'BAD_KEY' : 'NO_KEY'});
}

// Main AI call with session history
async function callAI(sys, msg, maxTokens=800, temp=0.82, sessionHistory=[]) {
  const messages=[{role:'system',content:sys}];
  for(const t of sessionHistory) messages.push({role:t.role==='model'?'assistant':t.role,content:t.text});
  messages.push({role:'user',content:msg});
  return enqueue(()=>route(messages,{maxTokens,temp}));
}

// Extra Gemini vision model — 1.5-flash has excellent free-tier vision support
const GEMINI_EP_15F = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';

// Vision call — image analysis (Gemini 2.0-flash → 2.0-flash-lite → 1.5-flash → OpenAI)
async function callVisionGeminiEP(ep, sys, b64, mime, userTxt, maxTokens=1200) {
  const body={
    contents:[{role:'user',parts:[
      {inlineData:{mimeType:mime,data:b64}},
      {text:userTxt}
    ]}],
    systemInstruction:{role:'system',parts:[{text:sys}]},
    generationConfig:{maxOutputTokens:maxTokens,temperature:0.4}
  };
  const ctrl=new AbortController();
  const tid=setTimeout(()=>ctrl.abort(),TIMEOUT_MS);
  let res,raw;
  try {
    res=await fetch(`${ep}?key=${GEMINI_KEY}`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body),signal:ctrl.signal
    });
    clearTimeout(tid); raw=await res.text();
  } catch(e){ clearTimeout(tid); if(e.name==='AbortError') throw new Error('Timed out'); throw e; }
  if(res.status===429){ throw new Error('Vision rate limited (429) — try again in a moment'); }
  if(!res.ok){ let m=raw; try{m=JSON.parse(raw)?.error?.message||raw;}catch(_){} throw new Error('Vision '+res.status+': '+m.slice(0,120)); }
  const data=JSON.parse(raw);
  if(data?.candidates?.[0]?.finishReason==='SAFETY') throw new Error('Image blocked by safety filter');
  const txt=data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if(!txt){ log.warn('Gemini vision empty, raw: '+raw.slice(0,200)); throw new Error('Vision returned empty — try a clearer image'); }
  return txt.trim();
}

// Vision has its own queue so it never blocks/is blocked by chat requests
// Vision queue — serializes ALL vision calls so Gemini never gets concurrent requests
// Gemini free tier: 15 RPM = 1 per 4s minimum. Queue ensures this is respected
// even when user taps Analyze rapidly.
const VQ = {
  queue: [],
  running: false,
  lastAt: 0,
  minGap: 4500, // 4.5s between calls = ~13 RPM, well under 15 RPM limit
};

function visionEnqueue(fn) {
  return new Promise((resolve, reject) => {
    VQ.queue.push({ fn, resolve, reject });
    visionDrainQueue();
  });
}

async function visionDrainQueue() {
  if(VQ.running || VQ.queue.length === 0) return;
  VQ.running = true;
  while(VQ.queue.length > 0) {
    const gap = VQ.minGap - (Date.now() - VQ.lastAt);
    if(gap > 0) await new Promise(r=>setTimeout(r, gap));
    const { fn, resolve, reject } = VQ.queue.shift();
    VQ.lastAt = Date.now();
    try { resolve(await fn()); } catch(e) { reject(e); }
  }
  VQ.running = false;
}

async function visionDirect(ep, sys, b64, mime, userTxt, maxTokens) {
  return visionEnqueue(() => callVisionGeminiEP(ep, sys, b64, mime, userTxt, maxTokens));
}

// HF Vision — BLIP not available on new HF router (hf-inference only supports
// CPU tasks like embeddings/classification, not image-to-text as of 2025)
// HF_KEY is still used for Kokoro TTS. Vision uses Gemini/OpenAI cascade.
async function callVisionHF(b64, mime, userTxt, maxTokens=600) {
  throw new Error('HF BLIP vision not available on current HF inference router');
}


async function callVision(sys, b64, mime, userTxt, maxTokens=800) {
  // === GEMINI CASCADE (free 15 RPM — primary vision provider) ===
  // HF BLIP removed: not supported on new hf-inference router (404).
  if(GEMINI_KEY && !BAD_KEYS.gemini) {
    const geminiModels = [
      { ep: GEMINI_EP_V2,  name: 'gemini-2.0-flash'      },
      { ep: GEMINI_EP_FB,  name: 'gemini-2.0-flash-lite'  },
      { ep: GEMINI_EP_15F, name: 'gemini-1.5-flash'       },
    ];
    let lastErr = null;
    for(const mdl of geminiModels) {
      // Brief wait after a rate-limit before next model
      if(lastErr && (lastErr.message.includes('429') || lastErr.message.includes('rate limit'))) {
        log.info('Vision rate limited on prev model — waiting 4s before '+mdl.name);
        await new Promise(r=>setTimeout(r, 4000));
      }
      try {
        const result = await visionDirect(mdl.ep, sys, b64, mime, userTxt, maxTokens);
        log.info('Vision OK via '+mdl.name);
        return result;
      } catch(e) {
        lastErr = e;
        log.warn('Vision '+mdl.name+' failed: '+e.message);
        // Hard auth failure — no point trying other Gemini models
        if(e.message.includes('401') || e.message.includes('403') ||
           e.message.includes('API_KEY_INVALID') || e.message.includes('API key')) {
          BAD_KEYS.gemini = true;
          log.warn('Gemini key invalid — disabling Gemini, trying OpenAI');
          break;
        }
      }
    }
    // All Gemini models failed — if it was rate-limit, say so clearly
    if(lastErr && !BAD_KEYS.gemini) {
      const isRL = lastErr.message.includes('429') || lastErr.message.includes('rate limit');
      if(isRL) throw Object.assign(
        new Error('Gemini vision rate limited — please wait 15 seconds and try again'),
        {code:'RATE_LIMITED'}
      );
      // Other transient failure — fall through to OpenAI if available
      log.warn('All Gemini models failed with non-auth error: '+lastErr.message);
    }
  }

  // === 3. OPENAI FALLBACK — only if key exists and not known bad ===
  if(!OPENAI_KEY || BAD_KEYS.openai) {
    const fixes = [];
    if(!GEMINI_KEY)        fixes.push('add GEMINI_API_KEY to .env (free at aistudio.google.com)');
    else if(BAD_KEYS.gemini) fixes.push('Gemini key invalid — check GEMINI_API_KEY in .env');
    if(!OPENAI_KEY)        fixes.push('or add OPENAI_API_KEY (paid, platform.openai.com)');
    else if(BAD_KEYS.openai) fixes.push('OpenAI key invalid — check OPENAI_API_KEY in .env');
    throw Object.assign(new Error('Vision unavailable: '+fixes.join(' | ')), {code:'VISION_NO_PROVIDER'});
  }
  const oaiMessages=[
    {role:'system',content:sys},
    {role:'user',content:[
      {type:'image_url',image_url:{url:`data:${mime};base64,${b64}`,detail:'low'}},
      {type:'text',text:userTxt}
    ]}
  ];
  const ctrl=new AbortController();
  const tid=setTimeout(()=>ctrl.abort(),30000);
  try {
    const res=await fetch(OPENAI_EP,{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+OPENAI_KEY},body:JSON.stringify({model:'gpt-4o-mini',messages:oaiMessages,max_tokens:maxTokens,temperature:0.4}),signal:ctrl.signal});
    clearTimeout(tid);
    if(res.status===401){
      BAD_KEYS.openai=true;
      throw Object.assign(new Error('OpenAI API key is invalid — check OPENAI_API_KEY in your .env file'), {code:'BAD_KEY'});
    }
    if(res.status===429){ throw Object.assign(new Error('Vision rate limited — try again in 30 seconds'), {code:'RATE_LIMITED'}); }
    if(!res.ok){ const t=await res.text().catch(()=>''); let m=''; try{m=JSON.parse(t)?.error?.message||t;}catch(_){m=t;} throw new Error('Vision error ('+res.status+'): '+m.slice(0,80)); }
    const d=await res.json();
    const txt=d.choices?.[0]?.message?.content;
    if(!txt) throw new Error('Vision returned empty response — try a clearer image');
    log.info('OpenAI vision OK ('+txt.length+' chars)');
    return txt.trim();
  } catch(e){ clearTimeout(tid); throw e; }
}

function fallback(type) {
  return {
    chat:"I'm having a moment of quiet — try again in a second.",
    story:'[Story engine temporarily unavailable]',
    vision:'[Vision temporarily unavailable]',
    voice:'[Voice temporarily unavailable]',
  }[type]||'[Temporarily unavailable]';
}

// Background user model update (non-blocking)
// Runs every 3 interactions — extracts rich structured data so buildPrompt
// can personalise every response with what ECHOAI knows about this person.
const MODEL_UPDATE_EVERY = 3;
async function updateUserModelBG(uid, userMsg, aiReply) {
  try {
    const {data, profile} = gp(uid);
    profile.interactionCount = (profile.interactionCount || 0) + 1;
    if(profile.interactionCount % MODEL_UPDATE_EVERY !== 0) {
      data.profiles[uid] = profile; wd(data); return;
    }
    const um = profile.userModel || {};
    // Rich extraction prompt — captures personality, style, and context
    const sys = 'You are a precise data extractor. Extract structured user information from a conversation snippet. Respond ONLY with valid minified JSON — no markdown, no explanation.';
    const existing = {
      interests:             um.interests?.slice(0, 6)         || [],
      keyFacts:              um.keyFacts?.slice(0, 5)          || [],
      lifeContext:           um.lifeContext                    || '',
      communicationStyle:    um.communicationStyle             || '',
      preferredResponseLength: um.preferredResponseLength      || '',
      currentFocus:          um.currentFocus                  || '',
    };
    const prompt = `Conversation:
User: "${userMsg.slice(0, 400)}"
AI: "${aiReply.slice(0, 300)}"

Existing model: ${JSON.stringify(existing)}

Return JSON: {"newInterests":[],"newKeyFacts":[],"lifeContext":"","communicationStyle":"","preferredResponseLength":"short|medium|detailed or ''","currentFocus":"","emotion":"neutral|positive|negative|curious|anxious|sad|frustrated|excited|creative"}`;
    const raw = await callAI(sys, prompt, 280, 0.2, []);
    let ex;
    try { ex = JSON.parse(raw.replace(/```json|```/g, '').trim()); } catch(_) { return; }
    // Merge interests — deduplicated, capped at 25
    if(ex.newInterests?.length)
      um.interests = [...new Set([...(um.interests || []), ...ex.newInterests])].slice(0, 25);
    // Merge key facts — deduplicated, capped at 20
    if(ex.newKeyFacts?.length)
      um.keyFacts = [...new Set([...(um.keyFacts || []), ...ex.newKeyFacts])].slice(0, 20);
    // Always update if non-empty (override with richer description)
    if(ex.lifeContext)              um.lifeContext           = ex.lifeContext;
    if(ex.communicationStyle)      um.communicationStyle    = ex.communicationStyle;
    if(ex.preferredResponseLength) um.preferredResponseLength = ex.preferredResponseLength;
    if(ex.currentFocus)            um.currentFocus          = ex.currentFocus;
    // Emotional arc — track last 100 data points
    if(ex.emotion) {
      profile.emotionalArc = profile.emotionalArc || [];
      profile.emotionalArc.push({ emotion: ex.emotion, ts: new Date().toISOString() });
      if(profile.emotionalArc.length > 100) profile.emotionalArc = profile.emotionalArc.slice(-100);
    }
    um.lastUpdated = new Date().toISOString();
    profile.userModel = um;
    data.profiles[uid] = profile;
    wd(data);
    log.debug('UserModel updated', { uid, interests: um.interests?.length, keyFacts: um.keyFacts?.length });
  } catch(e) { log.debug('User model update failed', { e: e.message }); }
}

// =============================================================================
// RESPONSE HELPERS
// =============================================================================
function ok(uid, msg, data, xpD) {
  const {profile}=gp(uid);
  const tier=getTier(profile.level);
  const range=getRange(profile.level);
  return {status:'success',message:msg,level:profile.level,xp:profile.xp,tier:tier.name,
    leveled_up:xpD?.leveled||false,old_level:xpD?.oldLevel||profile.level,
    xp_earned:xpD?(xpD.xp-(xpD.oldXp||0)):0,xp_range:range,data};
}
function err(msg, code=400) { return {status:'error',message:msg,code,data:null}; }

// =============================================================================
// EXPRESS + WEBSOCKET
// =============================================================================
const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({server});
const wsClients = new Map();

app.use(express.json({limit:'25mb'}));
app.use(express.urlencoded({extended:true}));
app.use((req,res,next)=>{
  const origin=req.headers.origin||'*';
  res.setHeader('Access-Control-Allow-Origin', origin==='*'?'*':origin);
  res.setHeader('Access-Control-Allow-Methods','GET,POST,PUT,DELETE,PATCH,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers','Content-Type,Authorization,X-Auth-Token,Accept');
  res.setHeader('Access-Control-Expose-Headers','X-TTS-Provider,X-TTS-Voice,Content-Type');
  res.setHeader('Access-Control-Allow-Credentials','true');
  res.setHeader('Access-Control-Max-Age','86400');
  if(req.method==='OPTIONS'){res.sendStatus(204);return;}
  res.setHeader('X-Content-Type-Options','nosniff');
  next();
});

const hits=new Map();
// Clean up stale rate-limit keys every 5 minutes to prevent memory leak
setInterval(()=>{ const now=Math.floor(Date.now()/60000); hits.forEach((_,k)=>{ if(!k.endsWith(':'+now)&&!k.endsWith(':'+(now-1))) hits.delete(k); }); }, 300000);
function rl(req,res,next){
  const ip=req.ip||'x'; const k=`${ip}:${Math.floor(Date.now()/60000)}`;
  hits.set(k,(hits.get(k)||0)+1);
  if(hits.get(k)>120) return res.status(429).json(err('Too many requests',429));
  next();
}
function auth(req,res,next){
  const tok=(req.headers.authorization||'').replace('Bearer ','')||req.headers['x-auth-token']||'';
  const p=verifyJWT(tok);
  if(!p) return res.status(401).json(err('Not authenticated',401));
  req.uid=p.uid; req.uname=p.name; next();
}

// WebSocket
wss.on('connection',(ws)=>{
  let uid=null;
  ws.on('message',(raw)=>{
    try {
      const msg=JSON.parse(raw);
      if(msg.type==='auth'){
        const p=verifyJWT(msg.token);
        if(!p){ws.send(JSON.stringify({type:'error',message:'Invalid token'}));return;}
        uid=p.uid; wsClients.set(uid,ws);
        ws.send(JSON.stringify({type:'authenticated',uid}));
        log.info('WS connected: '+p.name);
      }
      if(msg.type==='ping') ws.send(JSON.stringify({type:'pong'}));
    } catch(_){}
  });
  ws.on('close',()=>{ if(uid && wsClients.get(uid)===ws) wsClients.delete(uid); });
  ws.on('error',()=>{ if(uid && wsClients.get(uid)===ws) wsClients.delete(uid); });
});
function wsPush(uid,data){
  const ws=wsClients.get(uid);
  if(ws&&ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify(data));
}

// ── BROADCAST to every authenticated browser client ──────────────────────────
function wsBroadcast(data){
  const raw=JSON.stringify(data);
  for(const ws of wsClients.values()){
    if(ws.readyState===WebSocket.OPEN) ws.send(raw);
  }
}

// ── SENTINEL + PHANTOM BRIDGE ────────────────────────────────────────────────
// server.js connects as a WS *client* to both Python servers and re-broadcasts
// every incoming message to all authenticated browser clients. This is the
// missing link: Sentinel/Phantom → server.js → browser (index.html).
// ─────────────────────────────────────────────────────────────────────────────
const BRIDGE_SOURCES = [
  { name:'Sentinel', url:'ws://localhost:8767' },
  { name:'Phantom',  url:'ws://localhost:8766' },
];

function startBridge({name,url}){
  let ws=null, retryDelay=2000, retryTimer=null;

  function connect(){
    try {
      // perMessageDeflate:false — avoids compression negotiation issues with
      // some Python websockets versions on Android. Keepalive ping every 15s
      // prevents Android TCP idle-connection termination.
      ws = new WebSocket(url, { perMessageDeflate: false });

      let pingTimer = null;

      ws.on('open', ()=>{
        log.info(`Bridge [${name}] connected to ${url}`);
        retryDelay=2000;
        // Keep-alive ping every 15s (Android kills idle TCP after ~30s)
        pingTimer = setInterval(()=>{
          if(ws.readyState===WebSocket.OPEN) ws.ping();
        }, 15000);
        // Ask for all current state immediately
        try{ ws.send(JSON.stringify({type:'get_all'})); }catch(_){}
      });

      ws.on('message', (raw)=>{
        try {
          const msg = JSON.parse(raw);
          // Tag the message with its source so the frontend knows who sent it
          msg._bridge = name.toLowerCase();
          wsBroadcast(msg);
        } catch(_){}
      });

      ws.on('close', ()=>{
        clearInterval(pingTimer);
        log.warn(`Bridge [${name}] disconnected — retrying in ${retryDelay}ms`);
        retryTimer = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay*1.5, 30000);
      });

      ws.on('error', (e)=>{
        clearInterval(pingTimer);
        log.debug(`Bridge [${name}] error: ${e.message}`);
        // 'close' event fires after error, which handles reconnect
      });

    } catch(e){
      log.warn(`Bridge [${name}] connect failed: ${e.message}`);
      retryTimer = setTimeout(connect, retryDelay);
    }
  }

  // Delay first connect by 3s to let Python servers finish booting
  setTimeout(connect, 3000);
  return { stop(){ clearTimeout(retryTimer); ws&&ws.close(); } };
}

// Start both bridges when server starts
const bridges = BRIDGE_SOURCES.map(startBridge);
log.info('Sentinel+Phantom bridges initialised — will connect once Python servers are ready');

// =============================================================================
// ROOT ROUTE — serve index.html with auto-injected tunnel URL
// =============================================================================
let TUNNEL_URL = process.env.TUNNEL_URL || '';

function injectTunnelUrl(html, base) {
  return html.replace('<head>', `<head>\n<script>window.ECHOAI_BASE_URL="${base}";</script>`);
}

app.get('/',(req,res)=>{
  const f=path.join(__dirname,'index.html');
  if(!fs.existsSync(f)){
    res.status(404).send('<h2>index.html not found</h2><p>Put index.html in the same folder as server.js</p>');
    return;
  }
  let html=fs.readFileSync(f,'utf8');
  const fwdProto=req.headers['x-forwarded-proto'];
  const fwdHost=req.headers['x-forwarded-host']||req.headers['host'];
  if(!TUNNEL_URL && fwdProto==='https' && fwdHost) TUNNEL_URL='https://'+fwdHost;
  if(TUNNEL_URL) html=injectTunnelUrl(html,TUNNEL_URL);
  res.setHeader('Content-Type','text/html; charset=utf-8');
  res.setHeader('Cache-Control','no-cache');
  res.send(html);
});

// Static files
app.use(express.static(__dirname,{index:false}));

// manifest.json with PWA meta
app.get('/manifest.json',(req,res)=>{
  res.json({name:'ECHOAI',short_name:'ECHOAI',description:'Your self-evolving AI companion',
    start_url:'/',scope:'/',display:'standalone',background_color:'#050508',theme_color:'#00e5ff',
    orientation:'portrait-primary',icons:[
      {src:'/icon-192.png',sizes:'192x192',type:'image/png',purpose:'any maskable'},
      {src:'/icon-512.png',sizes:'512x512',type:'image/png',purpose:'any maskable'}
    ]});
});

// =============================================================================
// AUTH ROUTES
// =============================================================================
app.post('/api/auth/signup',rl,(req,res)=>{
  try {
    const {name,email,password}=req.body;
    if(!name||name.trim().length<2) return res.status(400).json(err('Name must be at least 2 characters'));
    if(!email||!email.includes('@')) return res.status(400).json(err('Valid email required'));
    if(!password||password.length<6) return res.status(400).json(err('Password must be at least 6 characters'));
    const d=rd(); const ek=email.trim().toLowerCase();
    if(Object.values(d.accounts).find(a=>a.email===ek)) return res.status(409).json(err('Email already registered'));
    const salt=crypto.randomBytes(16).toString('hex');
    const hash=crypto.createHash('sha256').update(salt+password).digest('hex');
    const uid='u_'+crypto.randomBytes(12).toString('hex');
    d.accounts[uid]={uid,name:name.trim(),email:ek,salt,hash,created:new Date().toISOString()};
    d.profiles[uid]=emptyProfile(); wd(d);
    const token=signJWT({uid,name:name.trim(),email:ek});
    log.info('Signup',{uid,name:name.trim()});
    res.status(201).json({status:'success',message:'Welcome to ECHOAI!',token,uid,name:name.trim(),email:ek,level:1,xp:0,tier:'Nascent Spark',avatar:'⚡',mode:'general'});
  } catch(e){ log.error('signup',{e:e.message}); res.status(500).json(err('Signup failed')); }
});

app.post('/api/auth/login',rl,(req,res)=>{
  try {
    const {email,password}=req.body;
    if(!email||!password) return res.status(400).json(err('Email and password required'));
    const d=rd(); const ek=email.trim().toLowerCase();
    const acc=Object.values(d.accounts).find(a=>a.email===ek);
    if(!acc) return res.status(401).json(err('No account with this email'));
    const hash=crypto.createHash('sha256').update(acc.salt+password).digest('hex');
    if(hash!==acc.hash) return res.status(401).json(err('Incorrect password'));
    const {profile}=gp(acc.uid);
    const token=signJWT({uid:acc.uid,name:acc.name,email:acc.email});
    log.info('Login',{uid:acc.uid});
    res.json({status:'success',message:`Welcome back, ${acc.name}!`,token,uid:acc.uid,name:acc.name,email:acc.email,level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,avatar:profile.avatar||'⚡',mode:profile.mode||'general'});
  } catch(e){ log.error('login',{e:e.message}); res.status(500).json(err('Login failed')); }
});

app.get('/api/auth/me',auth,(req,res)=>{
  try {
    const {profile}=gp(req.uid); const d=rd(); const acc=d.accounts[req.uid];
    res.json({status:'success',uid:req.uid,name:req.uname,email:acc?.email||'',level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,avatar:profile.avatar||'⚡',bio:profile.bio||'',mode:profile.mode||'general',hasPhoto:!!profile.profilePhoto});
  } catch(e){ res.status(500).json(err('Failed')); }
});

// =============================================================================
// PROFILE ROUTES
// =============================================================================
app.post('/api/profile/update',rl,auth,(req,res)=>{
  try {
    const {bio,avatar,mode,profilePhoto}=req.body;
    const {data,profile}=gp(req.uid);
    if(bio!==undefined) profile.bio=String(bio).slice(0,200);
    if(avatar!==undefined) profile.avatar=String(avatar).slice(0,4);
    if(mode!==undefined && MODES[mode]) profile.mode=mode;
    if(profilePhoto!==undefined){
      if(profilePhoto && profilePhoto.length>5.5*1024*1024) return res.status(400).json(err('Photo too large — max 4MB'));
      profile.profilePhoto=profilePhoto;
    }
    data.profiles[req.uid]=profile; wd(data);
    res.json({status:'success',message:'Profile updated',data:{bio:profile.bio,avatar:profile.avatar,mode:profile.mode,hasPhoto:!!profile.profilePhoto}});
  } catch(e){ res.status(500).json(err('Update failed')); }
});

app.get('/api/profile/:uid/photo',auth,(req,res)=>{
  try {
    const {profile}=gp(req.params.uid);
    if(!profile.profilePhoto) return res.status(404).json(err('No photo'));
    const match=profile.profilePhoto.match(/^data:([a-z/]+);base64,(.+)$/);
    if(!match) return res.status(400).json(err('Invalid photo'));
    const buf=Buffer.from(match[2],'base64');
    res.setHeader('Content-Type',match[1]);
    res.setHeader('Cache-Control','public, max-age=86400');
    res.send(buf);
  } catch(e){ res.status(500).json(err('Failed')); }
});

app.get('/api/profile/:uid',auth,(req,res)=>{
  try {
    const d=rd(); const acc=d.accounts[req.params.uid];
    if(!acc) return res.status(404).json(err('User not found'));
    const {profile}=gp(req.params.uid);
    res.json({status:'success',data:{uid:req.params.uid,name:acc.name,avatar:profile.avatar||'⚡',bio:profile.bio||'',level:profile.level,tier:getTier(profile.level).name,joined:profile.joined||acc.created,hasPhoto:!!profile.profilePhoto}});
  } catch(e){ res.status(500).json(err('Failed')); }
});

// =============================================================================
// FRIENDS ROUTES
// =============================================================================
app.post('/api/friends/request',rl,auth,(req,res)=>{
  try {
    const {toUid}=req.body;
    if(!toUid||toUid===req.uid) return res.status(400).json(err('Invalid user'));
    const d=rd();
    if(!d.accounts[toUid]) return res.status(404).json(err('User not found'));
    const {data,profile}=gp(req.uid);
    const {profile:tp}=gp(toUid);
    if(profile.friends[toUid]) return res.status(409).json(err('Already connected'));
    if(tp.friendRequests[req.uid]) return res.status(409).json(err('Request already sent'));
    tp.friendRequests[req.uid]={from:req.uid,fromName:req.uname,ts:new Date().toISOString()};
    data.profiles[toUid]=tp; wd(data);
    wsPush(toUid,{type:'friend_request',from:req.uid,fromName:req.uname});
    const xpD=awardXP(req.uid,XP.social);
    res.json({status:'success',message:'Friend request sent',data:{to:toUid},level:gp(req.uid).profile.level,xp:gp(req.uid).profile.xp,tier:getTier(gp(req.uid).profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel});
  } catch(e){ log.error('friend request',{e:e.message}); res.status(500).json(err('Failed')); }
});

app.post('/api/friends/respond',rl,auth,(req,res)=>{
  try {
    const {fromUid,accept}=req.body;
    const {data,profile}=gp(req.uid);
    if(!profile.friendRequests[fromUid]) return res.status(404).json(err('Request not found'));
    delete profile.friendRequests[fromUid];
    if(accept){
      profile.friends[fromUid]={since:new Date().toISOString()};
      const {profile:fp}=gp(fromUid);
      fp.friends[req.uid]={since:new Date().toISOString()};
      data.profiles[fromUid]=fp;
      wsPush(fromUid,{type:'friend_accepted',by:req.uid,byName:req.uname});
    }
    data.profiles[req.uid]=profile; wd(data);
    const xpD=accept?awardXP(req.uid,XP.social):null;
    res.json({status:'success',message:accept?'Friend added':'Request declined',data:{accepted:!!accept},level:gp(req.uid).profile.level,xp:gp(req.uid).profile.xp,tier:getTier(gp(req.uid).profile.level).name,leveled_up:xpD?.leveled||false,old_level:xpD?.oldLevel||gp(req.uid).profile.level});
  } catch(e){ res.status(500).json(err('Failed')); }
});

app.get('/api/friends/list',auth,(req,res)=>{
  try {
    const {profile}=gp(req.uid); const d=rd();
    const friends=Object.keys(profile.friends||{}).map(uid=>{
      const acc=d.accounts[uid]; const fp=d.profiles[uid]||{};
      return {uid,name:acc?.name||'Unknown',avatar:fp.avatar||'⚡',level:fp.level||1,tier:getTier(fp.level||1).name,online:wsClients.has(uid),hasPhoto:!!fp.profilePhoto};
    });
    const requests=Object.entries(profile.friendRequests||{}).map(([uid,r])=>({uid,name:r.fromName,ts:r.ts}));
    res.json({status:'success',data:{friends,requests}});
  } catch(e){ res.status(500).json(err('Failed')); }
});

app.delete('/api/friends/:uid',rl,auth,(req,res)=>{
  try {
    const {data,profile}=gp(req.uid);
    delete profile.friends[req.params.uid];
    const {profile:fp}=gp(req.params.uid);
    delete fp.friends[req.uid];
    data.profiles[req.uid]=profile; data.profiles[req.params.uid]=fp; wd(data);
    res.json({status:'success',message:'Removed'});
  } catch(e){ res.status(500).json(err('Failed')); }
});

app.get('/api/users/search',auth,(req,res)=>{
  try {
    const q=(req.query.q||'').toLowerCase().trim();
    if(q.length<2) return res.status(400).json(err('Query too short'));
    const d=rd();
    const results=Object.values(d.accounts)
      .filter(a=>a.uid!==req.uid&&a.name.toLowerCase().includes(q))
      .slice(0,10)
      .map(a=>{const p=d.profiles[a.uid]||{};return {uid:a.uid,name:a.name,avatar:p.avatar||'⚡',level:p.level||1,tier:getTier(p.level||1).name};});
    res.json({status:'success',data:{results}});
  } catch(e){ res.status(500).json(err('Search failed')); }
});

// =============================================================================
// MESSAGES ROUTES
// =============================================================================
app.post('/api/messages/send',rl,auth,(req,res)=>{
  try {
    const {toUid,text,msgType='text',mediaData,fileName}=req.body;
    if(!toUid) return res.status(400).json(err('toUid required'));
    if(msgType==='text'&&(!text||!text.trim())) return res.status(400).json(err('text required'));
    if((msgType==='image'||msgType==='voice'||msgType==='file')&&!mediaData) return res.status(400).json(err('mediaData required'));
    if(mediaData&&mediaData.length>12*1024*1024) return res.status(400).json(err('File too large — max 8MB'));
    const {data,profile}=gp(req.uid);
    if(!profile.friends[toUid]) return res.status(403).json(err('You must be friends to message'));
    const msg={
      id:'msg_'+Date.now()+crypto.randomBytes(3).toString('hex'),
      from:req.uid,fromName:req.uname,
      text:(text||'').trim().slice(0,1000),
      msgType,mediaData:mediaData||null,fileName:(fileName||'').slice(0,100),
      ts:new Date().toISOString(),read:false,
    };
    if(!profile.messages[toUid]) profile.messages[toUid]=[];
    profile.messages[toUid].push(msg);
    if(profile.messages[toUid].length>MAX_MSG) profile.messages[toUid]=profile.messages[toUid].slice(-MAX_MSG);
    const {profile:tp}=gp(toUid);
    if(!tp.messages[req.uid]) tp.messages[req.uid]=[];
    tp.messages[req.uid].push(msg);
    if(tp.messages[req.uid].length>MAX_MSG) tp.messages[req.uid]=tp.messages[req.uid].slice(-MAX_MSG);
    data.profiles[req.uid]=profile; data.profiles[toUid]=tp; wd(data);
    wsPush(toUid,{type:'message',id:msg.id,from:req.uid,fromName:req.uname,text:msg.text,msgType,fileName,ts:msg.ts,read:false});
    const xpD=awardXP(req.uid,XP.social);
    res.json({status:'success',message:'Sent',data:{message:{...msg,mediaData:undefined}},level:gp(req.uid).profile.level,xp:gp(req.uid).profile.xp,tier:getTier(gp(req.uid).profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel});
  } catch(e){ log.error('msg send',{e:e.message}); res.status(500).json(err('Failed')); }
});

app.get('/api/messages/:friendUid',auth,(req,res)=>{
  try {
    const {data,profile}=gp(req.uid);
    const msgs=(profile.messages||{})[req.params.friendUid]||[];
    if(profile.messages[req.params.friendUid]) profile.messages[req.params.friendUid].forEach(m=>m.read=true);
    data.profiles[req.uid]=profile; wd(data);
    res.json({status:'success',data:{messages:msgs.slice(-100)}});
  } catch(e){ res.status(500).json(err('Failed')); }
});

// =============================================================================
// STATUS
// =============================================================================
app.get('/api/status',auth,(req,res)=>{
  try {
    const {profile}=gp(req.uid);
    const tier=getTier(profile.level);
    const range=getRange(profile.level);
    const prog=range.range>0?Math.round(((profile.xp-range.current)/range.range)*100):100;
    const up=process.uptime();
    res.json({status:'success',xp:profile.xp,level:profile.level,tier:tier.name,data:{
      api_connected:HAS_AI&&!OAQ.DEAD,
      api_provider:AI_PROVIDER,
      api_model:ACTIVE_MODEL,
      uptime:`${Math.floor(up/3600)}h ${Math.floor((up%3600)/60)}m`,
      memory_mb:Math.round(process.memoryUsage().rss/1024/1024),
      timestamp:new Date().toISOString(),
      evolution:{level:profile.level,tier:tier.name,xp:profile.xp,xp_progress_pct:prog,xp_range:range},
      memory_count:profile.memory?.length||0,
      friends_count:Object.keys(profile.friends||{}).length,
      mode:profile.mode||'general',
    }});
  } catch(e){ res.status(500).json(err('Status failed')); }
});

// =============================================================================
// CHAT
// =============================================================================
app.post('/api/chat',rl,auth,async(req,res)=>{
  try {
    const {message}=req.body;
    if(!message||!message.trim()) return res.status(400).json(err('message required'));
    const trimmed=message.trim().slice(0,2000);
    const sys=buildPrompt(req.uid,'chat',{},trimmed);
    const hist=getSessionHistory(req.uid);
    let reply,uf=false;
    try { reply=await callAI(sys,trimmed,800,0.85,hist); }
    catch(e){
      log.warn('Chat AI failed: '+e.message);
      if(e.code==='BAD_KEY'||e.code==='NO_KEY') {
        reply='[KEY ERROR] '+e.message.slice(0,120)+'\n\nCheck your .env file and restart the server.';
      } else if(e.code==='DAILY_QUOTA') {
        reply="I'm rate-limited right now — try again in a moment!";
      } else {
        reply=fallback('chat');
      }
      uf=true;
    }
    addToSession(req.uid,'user',trimmed);
    addToSession(req.uid,'model',reply);
    const xpD=awardXP(req.uid,XP.chat);
    storeMem(req.uid,'chat',trimmed,reply);
    if(!uf) updateUserModelBG(req.uid,trimmed,reply).catch(()=>{});
    const {profile}=gp(req.uid);
    res.json({status:'success',level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(profile.level),data:{reply,tone:detectTone(trimmed+' '+reply),used_fallback:uf,session_turns:getSessionHistory(req.uid).length/2|0}});
  } catch(e){ log.error('chat',{e:e.message}); res.status(500).json(err('Chat failed')); }
});

// =============================================================================
// STORY
// =============================================================================
app.post('/api/story',rl,auth,async(req,res)=>{
  try {
    const {prompt,genre='fantasy'}=req.body;
    if(!prompt||!prompt.trim()) return res.status(400).json(err('prompt required'));
    const sg=['fantasy','sci-fi','mystery','horror','romance','adventure','thriller','cosmic'].includes(genre)?genre:'fantasy';
    const sys=buildPrompt(req.uid,'story',{genre:sg});
    let story,uf=false;
    try { story=await callAI(sys,`Write a gripping ${sg} story opening: "${prompt.trim().slice(0,600)}"`,1000,0.92); }
    catch(e){ story=fallback('story'); uf=true; }
    const xpD=awardXP(req.uid,XP.story);
    storeMem(req.uid,'story',prompt.trim(),story);
    const {profile}=gp(req.uid);
    res.json({status:'success',level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(profile.level),data:{story,genre:sg,used_fallback:uf}});
  } catch(e){ res.status(500).json(err('Story failed')); }
});

// =============================================================================
// VISION (text description analysis)
// =============================================================================
app.post('/api/vision',rl,auth,async(req,res)=>{
  try {
    const {description,context=''}=req.body;
    if(!description||!description.trim()) return res.status(400).json(err('description required'));
    const t=description.trim().slice(0,1200);
    const sys=buildPrompt(req.uid,'vision');
    let analysis,uf=false;
    try { analysis=await callAI(sys,`Analyze: "${t}"${context.trim()?` Context: "${context.trim().slice(0,300)}"`:''}.`,800,0.88); }
    catch(e){ analysis=fallback('vision'); uf=true; }
    const xpD=awardXP(req.uid,XP.vision);
    storeMem(req.uid,'vision',t,analysis);
    const {profile}=gp(req.uid);
    res.json({status:'success',level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(profile.level),data:{analysis,tone:detectTone(t+' '+analysis),used_fallback:uf}});
  } catch(e){ res.status(500).json(err('Vision failed')); }
});

// =============================================================================
// VISUAL (Camera image analysis)
// =============================================================================
app.post('/api/visual',rl,auth,async(req,res)=>{
  try {
    const {image,prompt,mode='analyze'}=req.body;
    if(!image||image.length<50) return res.status(400).json(err('image required — send base64 data URL'));
    if(image.length>4*1024*1024) return res.status(400).json(err('Image too large — resize before sending'));
    // Extract real mime type from data URL
    const mimeMatch=image.match(/^data:(image\/[a-z+]+);base64,/);
    const mime=mimeMatch?mimeMatch[1]:'image/jpeg';
    const b64=mimeMatch?image.slice(mimeMatch[0].length):image.replace(/^data:image\/[a-z+]+;base64,/,'');
    const modePrompts={
      analyze:'Describe what you see in this image. Be clear and helpful.',
      solve  :'Solve the problem in this image. Show your steps simply.',
      read   :'Read and transcribe all text you can see in this image.',
      identify:'List and briefly describe the main things you see.',
      assist :'Look at this image and give helpful, practical advice.',
    };
    const sys='You are a helpful visual assistant. Give clear, friendly, concise answers. No jargon.\n\n'+(modePrompts[mode]||modePrompts.analyze);
    const userTxt=prompt?`User asks: "${prompt.trim().slice(0,500)}"`:'What do you see in this image?';
    let analysis;
    try {
      analysis=await callVision(sys,b64,mime,userTxt,800);
    } catch(e){
      log.error('Visual analysis failed: '+e.message);
      if(e.code==='VISION_NO_PROVIDER' || e.code==='BAD_KEY' || e.message.includes('invalid'))
        return res.status(503).json({status:'error',message:e.message,code:503,retryable:false});
      if(e.message.includes('rate limit')||e.message.includes('429')||e.code==='VISION_QUOTA'||e.code==='RATE_LIMITED') {
        res.setHeader('Retry-After','15');
        return res.status(429).json({status:'error',message:'Vision rate limited — Gemini free tier has a 15 req/min limit. Trying OpenAI backup...',code:429,retry_after:15,retryable:true});
      }
      if(e.code==='DAILY_QUOTA') return res.status(429).json(err('Daily quota reached — try again later',429));
      return res.status(502).json(err('Vision failed: '+e.message.slice(0,120),502));
    }
    const xpD=awardXP(req.uid,XP.vision);
    storeMem(req.uid,'visual',`[${mode}] ${userTxt}`,analysis.slice(0,400));
    const {profile}=gp(req.uid);
    res.json({status:'success',level:profile.level,xp:profile.xp,tier:getTier(profile.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(profile.level),data:{analysis,mode,tone:detectTone(analysis)}});
  } catch(e){ log.error('visual',{e:e.message}); res.status(500).json(err('Visual failed')); }
});

// =============================================================================
// MEMORY
// =============================================================================
app.post('/api/memory',rl,auth,(req,res)=>{
  try {
    const {action='retrieve',limit=50,content,type='note',query,confirm}=req.body;
    const {data,profile}=gp(req.uid);
    const mem=profile.memory||[];
    if(action==='retrieve') return res.json({status:'success',data:{memories:mem.slice().reverse().slice(0,Math.min(limit,200)),total:mem.length}});
    if(action==='search'){
      if(!query) return res.status(400).json(err('query required'));
      const q=query.toLowerCase();
      const r=mem.filter(m=>(m.input||'').toLowerCase().includes(q)||(m.output||'').toLowerCase().includes(q)).slice(-50).reverse();
      return res.json({status:'success',data:{results:r,total:r.length}});
    }
    if(action==='store'){
      if(!content||!content.trim()) return res.status(400).json(err('content required'));
      const e=storeMem(req.uid,type,content.trim(),'[stored]');
      const xpD=awardXP(req.uid,XP.memory);
      const {profile:p2}=gp(req.uid);
      return res.json({status:'success',message:'Saved',data:{memory:e},level:p2.level,xp:p2.xp,tier:getTier(p2.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(p2.level)});
    }
    if(action==='clear'){
      if(!confirm) return res.status(400).json(err('confirm required'));
      profile.memory=[]; data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Cleared',data:{cleared:true}});
    }
    res.status(400).json(err('Unknown action'));
  } catch(e){ res.status(500).json(err('Memory failed')); }
});

// =============================================================================
// REMINDERS
// =============================================================================
app.post('/api/reminder',rl,auth,(req,res)=>{
  try {
    const {action='list',text,time,priority='normal',id,show_completed=false}=req.body;
    const {data,profile}=gp(req.uid);
    if(!profile.reminders) profile.reminders=[];
    if(action==='create'){
      if(!text||!text.trim()) return res.status(400).json(err('text required'));
      const r={id:'rem_'+Date.now()+crypto.randomBytes(3).toString('hex'),text:text.trim().slice(0,500),time:(time||'').slice(0,100),priority:['high','normal','low'].includes(priority)?priority:'normal',created:new Date().toISOString(),completed:false};
      profile.reminders.push(r); data.profiles[req.uid]=profile; wd(data);
      const xpD=awardXP(req.uid,XP.reminder);
      const {profile:p2}=gp(req.uid);
      return res.json({status:'success',message:'Reminder created',data:{reminder:r},level:p2.level,xp:p2.xp,tier:getTier(p2.level).name,leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_range:getRange(p2.level)});
    }
    if(action==='list'){
      const list=profile.reminders.filter(r=>show_completed||!r.completed).sort((a,b)=>({high:0,normal:1,low:2}[a.priority]||1)-({high:0,normal:1,low:2}[b.priority]||1));
      return res.json({status:'success',data:{reminders:list}});
    }
    if(action==='complete'){
      const r=profile.reminders.find(x=>x.id===id);
      if(!r) return res.status(404).json(err('Not found'));
      r.completed=true; r.completedAt=new Date().toISOString();
      data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Done',data:{reminder:r}});
    }
    if(action==='delete'){
      const i=profile.reminders.findIndex(x=>x.id===id);
      if(i===-1) return res.status(404).json(err('Not found'));
      profile.reminders.splice(i,1); data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Deleted',data:{}});
    }
    res.status(400).json(err('Unknown action'));
  } catch(e){ res.status(500).json(err('Reminder failed')); }
});

// =============================================================================
// IoT TRACKER
// =============================================================================
app.post('/api/iot',rl,auth,(req,res)=>{
  try {
    const {action='list',device,state,value,unit}=req.body;
    const {data,profile}=gp(req.uid);
    if(!profile.iot) profile.iot={};
    if(action==='update'){
      if(!device||!device.trim()) return res.status(400).json(err('device required'));
      const k=device.trim().toLowerCase().replace(/\s+/g,'_');
      const isNew=!profile.iot[k];
      profile.iot[k]={key:k,name:device.trim(),state:['on','off','standby','error'].includes(state)?state:'off',value:value!==undefined?String(value).slice(0,50):'',unit:(unit||'').slice(0,20),timestamp:new Date().toISOString()};
      data.profiles[req.uid]=profile; wd(data);
      const xpD=isNew?awardXP(req.uid,XP.iot):null;
      const {profile:p2}=gp(req.uid);
      return res.json({status:'success',message:'Updated',data:{device:profile.iot[k]},level:p2.level,xp:p2.xp,tier:getTier(p2.level).name,leveled_up:xpD?.leveled||false,old_level:xpD?.oldLevel||p2.level,xp_range:getRange(p2.level)});
    }
    if(action==='list') return res.json({status:'success',data:{devices:Object.values(profile.iot)}});
    if(action==='delete'){
      if(!device) return res.status(400).json(err('device required'));
      const k=device.toLowerCase().replace(/\s+/g,'_');
      if(!profile.iot[k]) return res.status(404).json(err('Not found'));
      delete profile.iot[k]; data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Removed',data:{}});
    }
    res.status(400).json(err('Unknown action'));
  } catch(e){ res.status(500).json(err('IoT failed')); }
});

// =============================================================================
// GOALS
// =============================================================================
app.post('/api/goals',rl,auth,(req,res)=>{
  try {
    const {action='list',text,category='personal',id,progress}=req.body;
    const {data,profile}=gp(req.uid);
    if(!profile.goals) profile.goals=[];
    if(action==='create'){
      if(!text||!text.trim()) return res.status(400).json(err('text required'));
      const g={id:'g_'+Date.now(),text:text.trim().slice(0,300),category:(['personal','fitness','learning','career','creative','financial','social'].includes(category)?category:'personal'),created:new Date().toISOString(),completed:false,progress:0};
      profile.goals.push(g); data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Goal set',data:{goal:g}});
    }
    if(action==='list') return res.json({status:'success',data:{goals:profile.goals,active:profile.goals.filter(g=>!g.completed).length}});
    if(action==='progress'){
      const goal=profile.goals.find(g=>g.id===id);
      if(!goal) return res.status(404).json(err('Not found'));
      goal.progress=Math.min(100,Math.max(0,parseInt(progress)||0));
      if(goal.progress===100) goal.completed=true;
      data.profiles[req.uid]=profile; wd(data);
      const xpD=goal.completed?awardXP(req.uid,50):null;
      const {profile:p2}=gp(req.uid);
      return res.json({status:'success',message:goal.completed?'Goal completed! +50 XP':'Progress updated',data:{goal},level:p2.level,xp:p2.xp,tier:getTier(p2.level).name,leveled_up:xpD?.leveled||false,old_level:xpD?.oldLevel||p2.level,xp_range:getRange(p2.level)});
    }
    if(action==='delete'){
      const i=profile.goals.findIndex(g=>g.id===id);
      if(i===-1) return res.status(404).json(err('Not found'));
      profile.goals.splice(i,1); data.profiles[req.uid]=profile; wd(data);
      return res.json({status:'success',message:'Goal removed',data:{}});
    }
    res.status(400).json(err('Unknown action'));
  } catch(e){ res.status(500).json(err('Goals failed')); }
});

// =============================================================================
// INSIGHTS
// =============================================================================
app.get('/api/insights',auth,(req,res)=>{
  try {
    const {profile}=gp(req.uid);
    const insights=profile.insights||[];
    res.json({status:'success',data:{insights:insights.slice(-20).reverse(),total:insights.length}});
  } catch(e){ res.status(500).json(err('Failed')); }
});

app.post('/api/insights/generate',rl,auth,async(req,res)=>{
  try {
    const {profile}=gp(req.uid);
    const mem=profile.memory||[];
    if(mem.length<5) return res.json({status:'success',data:{insight:null,message:'Keep chatting — insights unlock after more interactions'}});
    const recent=mem.slice(-20).map(m=>`[${m.type}/${m.tone}] "${(m.input||'').slice(0,80)}"`).join('\n');
    const goals=(profile.goals||[]).map(g=>g.text).join(', ')||'none set';
    const um=profile.userModel||{};
    const sys=buildPrompt(req.uid,'insight');
    const prompt=`Recent interactions:\n${recent}\n\nKnown: interests=${(um.interests||[]).join(',')||'unknown'}, goals=${goals}\n\nGenerate one meaningful insight about their patterns or growth.`;
    let insightText;
    try { insightText=await callAI(sys,prompt,200,0.9,[]); }
    catch(e){ return res.json({status:'success',data:{insight:null,message:'Insight generation unavailable right now'}}); }
    const entry={id:'i_'+Date.now(),text:insightText,ts:new Date().toISOString(),memoryCount:mem.length};
    const {data,profile:p2}=gp(req.uid);
    if(!p2.insights) p2.insights=[];
    p2.insights.push(entry);
    if(p2.insights.length>50) p2.insights=p2.insights.slice(-50);
    data.profiles[req.uid]=p2; wd(data);
    res.json({status:'success',data:{insight:entry}});
  } catch(e){ res.status(500).json(err('Insights failed')); }
});

// =============================================================================
// ECHO PULSE
// =============================================================================
app.post('/api/pulse',rl,auth,async(req,res)=>{
  try {
    const {profile}=gp(req.uid);
    const um=profile.userModel||{};
    const goals=(profile.goals||[]).filter(g=>!g.completed).slice(0,3).map(g=>g.text).join(', ')||'none';
    const lastTone=(profile.emotionalArc||[]).slice(-1)[0]?.emotion||'neutral';
    const sys=buildPrompt(req.uid,'pulse');
    const prompt=`User: interests=${(um.interests||[]).slice(0,3).join(',')||'unknown'}, tone=${lastTone}, goals=${goals}. Send a meaningful daily pulse.`;
    let pulse;
    try { pulse=await callAI(sys,prompt,150,0.9,[]); }
    catch(e){ pulse="Today is a new opportunity. I'm here whenever you need me."; }
    res.json({status:'success',data:{pulse,tone:lastTone,timestamp:new Date().toISOString()}});
  } catch(e){ res.status(500).json(err('Pulse failed')); }
});

// =============================================================================
// DAILY REFLECTION
// =============================================================================
app.post('/api/reflect',rl,auth,async(req,res)=>{
  try {
    const {prompt}=req.body;
    const {profile}=gp(req.uid);
    const mem=profile.memory||[];
    const recent=mem.slice(-10).map(m=>`[${m.type}] "${(m.input||'').slice(0,60)}"`).join('\n')||'No recent activity';
    const sys=buildPrompt(req.uid,'reflection');
    const userPrompt=prompt?`Reflection prompt: "${prompt}"\n\nRecent activity:\n${recent}`:`Based on recent activity:\n${recent}\n\nGenerate a meaningful reflection question.`;
    let reflection;
    try { reflection=await callAI(sys,userPrompt,250,0.88,[]); }
    catch(e){ reflection="Take a moment to breathe. What's one thing you're grateful for today?"; }
    const xpD=awardXP(req.uid,15);
    const {data,profile:p2}=gp(req.uid);
    if(!p2.reflections) p2.reflections=[];
    p2.reflections.push({text:reflection,prompt:prompt||'',ts:new Date().toISOString()});
    if(p2.reflections.length>30) p2.reflections=p2.reflections.slice(-30);
    data.profiles[req.uid]=p2; wd(data);
    res.json({status:'success',xp:p2.xp,level:p2.level,tier:getTier(p2.level).name,
      leveled_up:xpD.leveled,old_level:xpD.oldLevel,xp_earned:15,xp_range:getRange(p2.level),
      data:{reflection,history:p2.reflections.slice(-5).reverse()}});
  } catch(e){ res.status(500).json(err('Reflection failed')); }
});

// =============================================================================
// USER MODEL
// =============================================================================
app.get('/api/usermodel',auth,(req,res)=>{
  try {
    const {profile}=gp(req.uid);
    res.json({status:'success',data:{userModel:profile.userModel||{},emotionalArc:(profile.emotionalArc||[]).slice(-20),interactionCount:profile.interactionCount||0}});
  } catch(e){ res.status(500).json(err('Failed')); }
});

// =============================================================================
// SESSION CLEAR
// =============================================================================
app.post('/api/session/clear',auth,(req,res)=>{
  clearSession(req.uid);
  res.json({status:'success',message:'Session cleared'});
});

// =============================================================================
// RESET QUOTA
// =============================================================================
app.post('/api/reset-quota',auth,(req,res)=>{
  OAQ.DEAD=false; OAQ.fails=0; OAQ.cooldown=0;
  res.json({status:'success',message:'Queue reset'});
});

// =============================================================================
// TTS — Real Human Voices
// Priority: Edge TTS (free, no key) → ElevenLabs → Azure → Groq Orpheus → OpenAI
// =============================================================================
const ELEVENLABS_KEY = process.env.ELEVENLABS_API_KEY || ''; // Optional: elevenlabs.io
const ELEVENLABS_EP  = 'https://api.elevenlabs.io/v1/text-to-speech';
let   EL_TIER           = '';
let   EL_BLOCKED        = false;  // true = confirmed VPN/unusual_activity block
let   EL_BLOCKED_REASON = '';     // human-readable reason shown in frontend
let   EL_NET_ERROR_AT   = 0;      // timestamp of last transient network error
// EL auto-retry: transient network errors reset after 10 minutes
// VPN/unusual_activity blocks are permanent for the session
function isELBlocked() {
  if(EL_BLOCKED) return true;                                       // hard block
  if(EL_NET_ERROR_AT && Date.now() - EL_NET_ERROR_AT < 600000) return true; // 10-min soft block
  return false;
}

// Fetch EL subscription tier once at startup — determines which models we can use
async function fetchELTier() {
  if(!ELEVENLABS_KEY || EL_TIER) return;
  try {
    const r = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
      headers: {'xi-api-key': ELEVENLABS_KEY}
    });
    if(!r.ok) return;
    const d = await r.json();
    EL_TIER = (d.tier || d.billing_tier || '').toLowerCase();
    log.info('ElevenLabs tier: '+(EL_TIER||'unknown'));
  } catch(e) {
    if(e.message.includes('fetch failed') || e.message.includes('ECONNREFUSED') || e.message.includes('ENOTFOUND') || e.message.includes('ETIMEDOUT')) {
      EL_NET_ERROR_AT = Date.now();
      EL_BLOCKED_REASON = 'ElevenLabs unreachable (network/firewall)';
      log.warn('EL tier unreachable — soft block 10min');
    } else {
      log.warn('EL tier fetch: '+e.message);
    }
  }
}

// Groq Orpheus TTS voices — canopylabs/orpheus-v1-english (replaced PlayAI Dec 2025)
const GROQ_TTS_VOICES = ['troy','austin','hannah','tara','leah','leo'];
const GROQ_TTS_MODEL  = 'canopylabs/orpheus-v1-english';

// ElevenLabs — seed voice IDs (name -> voice_id)
let EL_VOICES = {
  'Rachel':    '21m00Tcm4TlvDq8ikWAM',
  'Drew':      '29vD33N1CtxCmqQRPOHJ',
  'Clyde':     '2EiwWnXFnvU5JabPnv8n',
  'Domi':      'AZnzlk1XvdvUeBnXmlld',
  'Bella':     'EXAVITQu4vr4xnSDxMaL',
  'Antoni':    'ErXwobaYiN019PkySvjV',
  'Elli':      'MF3mGyEYCl7XYWbV9V6O',
  'Josh':      'TxGEqnHWrfWFTfGW9XjX',
  'Adam':      'pNInz6obpgDQGcFmaJgB',
  'Sam':       'yoZ06aMxZJJ28mfd3POQ',
  'Aria':      '9BWtsMINqrJLrRacOk9x',
  'Roger':     'CwhRBWXzGAHq8TQ4Fs17',
  'Laura':     'FGY2WhTYpPnrIDTdsKH5',
  'Charlie':   'IKne3meq5aSn9XLyUdCD',
  'George':    'JBFqnCBsd6RMkjVDRZzb',
  'Callum':    'N2lVS1w4EtoT3dr4eOWO',
  'Liam':      'TX3LPaxmHKxFdv7VOQHJ',
  'Charlotte': 'XB0fDUnXU5powFXDhCwa',
  'Alice':     'Xb7hH8MSUJpSbSDYk0k2',
  'Matilda':   'XrExE9yKIg1WjnnlVkGX',
  'Will':      'bIHbv24MWmeRgasZH58o',
  'Jessica':   'cgSgspJ2msm6clMCkdW9',
  'Eric':      'cjVigY5qzO86Huf0OWal',
  'Chris':     'iP95p4xoKVk53GoZ742B',
  'Brian':     'nPczCjzI2devNBz1zQrb',
  'Daniel':    'onwK4e9ZLuTAKqWW03F9',
  'Lily':      'pFZP5JQG7iQjIQuC4Bku',
  'Bill':      'pqHfZKP75CvOlQylNhV4',
};
// Full metadata from API: [{id, name, labels, description, category, preview_url}]
let EL_VOICE_META = [];
let EL_VOICES_FETCHED = false;

// Fetch real voice list from ElevenLabs API — called at startup and on demand
async function fetchELVoices() {
  if(!ELEVENLABS_KEY) return;
  try {
    const r = await fetch('https://api.elevenlabs.io/v1/voices', {
      headers: {'xi-api-key': ELEVENLABS_KEY}
    });
    if(!r.ok) { log.warn('EL voices fetch: HTTP '+r.status); return; }
    const data = await r.json();
    if(data.voices && data.voices.length > 0) {
      EL_VOICE_META = data.voices.map(v => ({
        id:          v.voice_id,
        name:        v.name,
        category:    v.category || 'premade',
        labels:      v.labels   || {},
        description: v.description || '',
        preview_url: v.preview_url || '',
      }));
      // Rebuild name→id map from live data
      const fresh = {};
      data.voices.forEach(v => { fresh[v.name] = v.voice_id; });
      EL_VOICES = { ...EL_VOICES, ...fresh };
      EL_VOICES_FETCHED = true;
      log.info('ElevenLabs: '+data.voices.length+' voices loaded');
    }
  } catch(e) {
    if(e.message.includes('fetch failed') || e.message.includes('ECONNREFUSED') || e.message.includes('ENOTFOUND') || e.message.includes('ETIMEDOUT')) {
      EL_NET_ERROR_AT = Date.now();
      EL_BLOCKED_REASON = 'ElevenLabs unreachable (network/firewall) — will auto-retry in 10 minutes';
      log.warn('EL unreachable — soft block for 10min: '+e.message.slice(0,60));
    } else {
      log.warn('EL voices fetch error: '+e.message);
    }
  }
}
// OpenAI TTS voices
const OAI_TTS_VOICES = ['alloy','echo','fable','onyx','nova','shimmer'];

app.get('/api/tts/voices', auth, async (req,res) => {
  if(ELEVENLABS_KEY && !EL_VOICES_FETCHED && !isELBlocked()) {
    await fetchELVoices().catch(()=>{});
  }
  const voices = [];
  if(ELEVENLABS_KEY && !isELBlocked()) {
    if(EL_VOICE_META.length > 0) {
      EL_VOICE_META.forEach(v => voices.push({
        id:          'el:' + v.id,      // ← actual voice_id, no lookup needed
        name:        v.name,
        provider:    'ElevenLabs',
        human:       true,
        category:    v.category  || 'premade',
        labels:      v.labels    || {},
        description: v.description || '',
        preview_url: v.preview_url  || '',
      }));
    } else {
      // Seed fallback
      Object.entries(EL_VOICES).forEach(([name, vid]) => voices.push({
        id: 'el:' + vid, name, provider:'ElevenLabs', human:true,
        category:'premade', labels:{}, description:'', preview_url:'',
      }));
    }
  }
  if(GROQ_KEY && !TTS_BAD.groq) {
    const groqDisplayNames = {troy:'Troy (US Male)',austin:'Austin (US Male)',hannah:'Hannah (US Female)',
      tara:'Tara (US Female)',leah:'Leah (US Female)',leo:'Leo (US Male)'};
    GROQ_TTS_VOICES.forEach(v => voices.push({id:'groq:'+v, name:groqDisplayNames[v]||v, provider:'Groq Orpheus', human:true}));
  }
  // Edge TTS — always available, no key required, Azure Neural quality
  if(!EDGE_TTS_BAD) {
    EDGE_TTS_VOICES.forEach(v => voices.push({id:v.id, name:v.name, provider:'Edge TTS (Free)', human:true}));
  }
  if(AZURE_TTS_KEY && !AZURE_TTS_BAD) {
    const azVoices = [
      {id:'azure:jenny', name:'Jenny (US Female)',    human:true},
      {id:'azure:aria',  name:'Aria (US Female)',     human:true},
      {id:'azure:guy',   name:'Guy (US Male)',        human:true},
      {id:'azure:davis', name:'Davis (US Male)',      human:true},
      {id:'azure:jane',  name:'Jane (US Female)',     human:true},
      {id:'azure:jason', name:'Jason (US Male)',      human:true},
      {id:'azure:nancy', name:'Nancy (US Female)',    human:true},
      {id:'azure:tony',  name:'Tony (US Male)',       human:true},
      {id:'azure:sonia', name:'Sonia (UK Female)',    human:true},
      {id:'azure:ryan',  name:'Ryan (UK Male)',       human:true},
      {id:'azure:libby', name:'Libby (UK Female)',    human:true},
    ];
    azVoices.forEach(v => voices.push({...v, provider:'Azure TTS'}));
  }
  if(HF_KEY && !KOKORO_BAD) {
    const kokoroVoices = [
      {id:'kokoro:af_heart',   name:'Heart (US Female)',   human:true},
      {id:'kokoro:af_bella',   name:'Bella (US Female)',   human:true},
      {id:'kokoro:am_adam',    name:'Adam (US Male)',      human:true},
      {id:'kokoro:am_michael', name:'Michael (US Male)',   human:true},
      {id:'kokoro:bf_emma',    name:'Emma (UK Female)',    human:true},
      {id:'kokoro:bm_george',  name:'George (UK Male)',    human:true},
    ];
    kokoroVoices.forEach(v => voices.push({...v, provider:'Kokoro (HF)'}));
  }
  if(OPENAI_KEY && !TTS_BAD.openai) {
    OAI_TTS_VOICES.forEach(v => voices.push({id:'oai:'+v, name:v[0].toUpperCase()+v.slice(1), provider:'OpenAI TTS', human:true}));
  }
  voices.push({id:'browser', name:'Browser (built-in)', provider:'Device', human:false});
  const hasTTS = true; // Edge TTS always available as fallback
  const active = (ELEVENLABS_KEY&&!isELBlocked()) ? 'ElevenLabs'
    : !EDGE_TTS_BAD ? 'Edge TTS'
    : (AZURE_TTS_KEY&&!AZURE_TTS_BAD) ? 'Azure TTS'
    : (GROQ_KEY&&!TTS_BAD.groq) ? 'Groq Orpheus'
    : (HF_KEY&&!KOKORO_BAD) ? 'Kokoro (HF)'
    : (OPENAI_KEY&&!TTS_BAD.openai) ? 'OpenAI' : 'browser';
  const elBlockedReason = isELBlocked() ? (EL_BLOCKED_REASON || (EL_NET_ERROR_AT ? 'Network error — retrying in 10 min' : '')) : '';
  res.json({status:'success', data:{voices, hasTTS, activeProvider:active, elBlocked:isELBlocked(), elBlockedReason}});
});

app.post('/api/tts', rl, auth, async (req,res) => {
  try {
    const {text, voiceId='auto'} = req.body;
    if(!text||!text.trim()) return res.status(400).json(err('text required'));
    const clean = text.trim().slice(0, 1000);

    // Helper to try ElevenLabs
    async function tryEL(vId) {
      if(!ELEVENLABS_KEY || isELBlocked()) return null;

      let vid, voiceName;
      if(vId && vId.startsWith('el:')) {
        vid = vId.slice(3);
        const meta = EL_VOICE_META.find(v => v.id === vid);
        voiceName = meta ? meta.name : vid.slice(0,8);
      } else {
        if(!EL_VOICES_FETCHED) await fetchELVoices().catch(()=>{});
        vid = Object.values(EL_VOICES)[0] || '21m00Tcm4TlvDq8ikWAM';
        voiceName = Object.keys(EL_VOICES)[0] || 'Rachel';
      }

      if(!vid) { log.warn('EL: no voice ID for '+vId); return null; }
      log.info('EL TTS: voice='+voiceName+' id='+vid+' tier='+(EL_TIER||'unknown'));

      // Try universally available models first — multilingual_v2 works on all tiers
      const models = ['eleven_multilingual_v2', 'eleven_flash_v2_5', 'eleven_turbo_v2_5'];

      for(const model_id of models) {
        const body = { text: clean, model_id, voice_settings: { stability: 0.5, similarity_boost: 0.75 } };
        const ctrl = new AbortController();
        const tid  = setTimeout(()=>ctrl.abort(), 25000);
        let r;
        try {
          r = await fetch(`${ELEVENLABS_EP}/${vid}?output_format=mp3_44100_128`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'xi-api-key': ELEVENLABS_KEY },
            body: JSON.stringify(body),
            signal: ctrl.signal
          });
          clearTimeout(tid);
        } catch(e) { clearTimeout(tid); log.warn('EL fetch ('+model_id+'): '+e.message); continue; }

        if(r.ok) {
          const audio = await r.arrayBuffer();
          if(audio.byteLength < 100) { log.warn('EL '+model_id+': suspiciously small response ('+audio.byteLength+'B) — skipping'); continue; }
          log.info('EL TTS OK: '+voiceName+' / '+model_id+' ('+audio.byteLength+' bytes)');
          return { audio, provider: 'elevenlabs', voice: voiceName };
        }

        const errText = await r.text().catch(()=>'');
        log.warn('EL '+model_id+' HTTP '+r.status+' body: '+errText.slice(0,300));
        if(r.status === 401) {
          let body401 = ''; try{ body401 = errText; } catch(_){}
          if(body401.includes('unusual_activity') || body401.includes('unusual activity') || body401.includes('Free Tier usage disabled')) {
            EL_BLOCKED = true;
            EL_BLOCKED_REASON = 'ElevenLabs blocked: Free tier disabled — VPN/proxy detected. Disable VPN or upgrade to paid plan.';
            log.warn('EL blocked: '+EL_BLOCKED_REASON);
            return null;
          }
          log.warn('EL 401 — invalid API key (check ELEVENLABS_API_KEY in .env)');
          return null;
        }
        if(r.status === 429) { log.warn('EL 429 — quota exhausted, try again later'); return null; }
        if(r.status === 422) { log.warn('EL 422 — bad request (model/voice/format issue): '+errText.slice(0,200)); continue; }
        if(r.status === 400) { log.warn('EL 400 — bad request: '+errText.slice(0,200)); continue; }
        // 400/403/422 = model not on this plan — try next
      }

      log.warn('EL: all models exhausted for voice '+voiceName);
      return null;
    }
    async function tryGroq(vId) {
      if(!GROQ_KEY || TTS_BAD.groq) return null;
      // Orpheus v1 English — replacement for decommissioned PlayAI (Dec 2025)
      // Voices: troy, austin, hannah, tara, leah, leo
      const reqVoice = vId && vId.startsWith('groq:') ? vId.slice(5) : null;
      // Map old PlayAI names to Orpheus equivalents
      const aliasMap = { 'Celeste-PlayAI':'hannah','Chip-PlayAI':'tara','Nia-PlayAI':'leah',
        'Fritz-PlayAI':'leo','Aaliyah-PlayAI':'troy','Adelaide-PlayAI':'austin' };
      const mappedVoice = reqVoice && aliasMap[reqVoice] ? aliasMap[reqVoice] : reqVoice;
      const voice = GROQ_TTS_VOICES.includes(mappedVoice) ? mappedVoice : 'troy';

      const body = { model: GROQ_TTS_MODEL, input: clean, voice, response_format: 'wav' };
      const ctrl = new AbortController();
      const tid  = setTimeout(()=>ctrl.abort(), 20000);
      let r;
      try {
        r = await fetch('https://api.groq.com/openai/v1/audio/speech', {
          method:'POST',
          headers:{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY},
          body:JSON.stringify(body), signal:ctrl.signal
        });
        clearTimeout(tid);
      } catch(e){ clearTimeout(tid); log.warn('Groq TTS fetch: '+e.message); return null; }

      if(r.status===401){ TTS_BAD.groq=true; log.warn('Groq TTS 401 — key invalid'); return null; }
      if(!r.ok){
        const t=await r.text().catch(()=>'');
        log.warn('Groq TTS HTTP '+r.status+': '+t.slice(0,120));
        if(t.includes('decommissioned')||t.includes('no longer supported')||t.includes('requires terms')||t.includes('terms acceptance')){
          TTS_BAD.groq=true; log.warn('Groq TTS permanently unavailable — disabling for session');
        }
        return null;
      }
      const audio = await r.arrayBuffer();
      if(audio.byteLength < 200){ log.warn('Groq TTS: tiny response '+audio.byteLength+'B'); return null; }
      log.info('Groq Orpheus TTS OK: voice='+voice+' ('+audio.byteLength+'B)');
      return {audio, provider:'groq-orpheus', voice, contentType:'audio/wav'};
    }

    // Helper to try OpenAI TTS
    async function tryOAI(vId) {
      if(!OPENAI_KEY || TTS_BAD.openai) return null;
      const voice = vId && vId.startsWith('oai:') ? vId.slice(4) : 'nova';
      const body  = { model:'tts-1', input:clean, voice, response_format:'mp3' };
      const ctrl  = new AbortController();
      const tid   = setTimeout(()=>ctrl.abort(), 20000);
      let r;
      try {
        r = await fetch('https://api.openai.com/v1/audio/speech', {
          method:'POST',
          headers:{'Content-Type':'application/json','Authorization':'Bearer '+OPENAI_KEY},
          body:JSON.stringify(body), signal:ctrl.signal
        });
        clearTimeout(tid);
      } catch(e){ clearTimeout(tid); log.warn('OAI TTS fetch: '+e.message); return null; }
      if(r.status===401){ TTS_BAD.openai=true; log.warn('OpenAI TTS 401 — key invalid, disabling for session'); return null; }
      if(!r.ok){ const t=await r.text().catch(()=>''); log.warn('OAI TTS '+r.status+': '+t.slice(0,80)); return null; }
      const audio = await r.arrayBuffer();
      if(audio.byteLength < 200){ log.warn('OAI TTS: tiny response '+audio.byteLength+'B'); return null; }
      log.info('OAI TTS OK: voice='+voice+' ('+audio.byteLength+'B)');
      return {audio, provider:'openai', voice};
    }

    // Helper: Microsoft Edge TTS — FREE, no key, no VPN issues, Azure Neural voices
    async function tryEdgeTTS(vId) {
      if(EDGE_TTS_BAD) return null;
      const voiceEntry = vId && vId.startsWith('edge:')
        ? EDGE_TTS_VOICES.find(v=>v.id===vId)
        : null;
      const voiceName = voiceEntry ? voiceEntry.voice : 'en-US-AriaNeural';

      // Generate Sec-MS-GEC token (SHA-256 of rounded timestamp + token)
      function makeGec() {
        const WIN_EPOCH = 11644473600n;
        const ticks = (BigInt(Date.now()) / 1000n + WIN_EPOCH) * 10000000n;
        const rounded = (ticks / 3000000000n) * 3000000000n;
        return require('crypto').createHash('sha256')
          .update(`${rounded}\n${EDGE_TTS_TOKEN.toUpperCase()}`)
          .digest('hex').toUpperCase();
      }

      return new Promise((resolve) => {
        const reqId  = crypto.randomBytes(16).toString('hex').toUpperCase();
        const secGec = makeGec();
        const url    = `${EDGE_TTS_WSS}?TrustedClientToken=${EDGE_TTS_TOKEN}&Sec-MS-GEC=${secGec}&Sec-MS-GEC-Version=1-130.0.2849.68&ConnectionId=${reqId}`;

        let ws, timer, settled = false;
        const chunks = [];

        function done(result) {
          if(settled) return; settled = true;
          clearTimeout(timer);
          try { if(ws) ws.terminate(); } catch(_) {}
          resolve(result);
        }

        try {
          ws = new WebSocket(url, {
            headers: {
              'User-Agent':    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0',
              'Origin':        'chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold',
              'Pragma':        'no-cache',
              'Cache-Control': 'no-cache',
            }
          });
        } catch(e) { log.warn('Edge TTS WS create error: '+e.message); return resolve(null); }

        timer = setTimeout(() => { log.warn('Edge TTS timeout (15s)'); done(null); }, 15000);

        ws.on('open', () => {
          const ts = new Date().toISOString();
          // Config message
          ws.send(`X-Timestamp:${ts}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n` +
            `{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},"outputFormat":"audio-24khz-96kbitrate-mono-mp3"}}}}`);
          // SSML message
          const escaped = clean.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&apos;').replace(/"/g,'&quot;');
          const ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='${voiceName}'><prosody rate='+0%' pitch='+0Hz'>${escaped}</prosody></voice></speak>`;
          ws.send(`X-RequestId:${reqId}\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:${ts}Z\r\nPath:ssml\r\n\r\n${ssml}`);
        });

        ws.on('message', (data) => {
          if(Buffer.isBuffer(data)) {
            // Binary frame — audio bytes follow the header block
            const sep = Buffer.from('\r\n\r\n');
            const idx = data.indexOf(sep);
            if(idx !== -1) chunks.push(data.slice(idx + 4));
          } else {
            const msg = String(data);
            if(msg.includes('Path:turn.end')) {
              const audio = Buffer.concat(chunks);
              if(audio.length > 200) {
                log.info('Edge TTS OK: voice='+voiceName+' ('+audio.length+'B)');
                done({audio, provider:'edge-tts', voice:voiceName, contentType:'audio/mpeg'});
              } else {
                log.warn('Edge TTS: turn.end but no audio ('+audio.length+'B)');
                done(null);
              }
            }
          }
        });

        ws.on('error', (e) => { log.warn('Edge TTS WS error: '+e.message); done(null); });
        ws.on('close', (code) => {
          if(!settled) {
            if(code === 1008 || code === 1003) { EDGE_TTS_BAD = true; log.warn('Edge TTS perm error code '+code); }
            done(null);
          }
        });
      });
    }

    // Helper: Azure Cognitive Services TTS — FREE 500K chars/month, 400+ voices
    async function tryAzure(vId) {
      if(!AZURE_TTS_KEY || AZURE_TTS_BAD) return null;
      // Voice name format: en-US-JennyNeural, en-GB-SoniaNeural, etc.
      const azureVoices = {
        'azure:jenny':  'en-US-JennyNeural',
        'azure:aria':   'en-US-AriaNeural',
        'azure:guy':    'en-US-GuyNeural',
        'azure:davis':  'en-US-DavisNeural',
        'azure:jane':   'en-US-JaneNeural',
        'azure:jason':  'en-US-JasonNeural',
        'azure:nancy':  'en-US-NancyNeural',
        'azure:tony':   'en-US-TonyNeural',
        'azure:sonia':  'en-GB-SoniaNeural',
        'azure:ryan':   'en-GB-RyanNeural',
        'azure:libby':  'en-GB-LibbyNeural',
      };
      const voiceName = (vId && azureVoices[vId]) || azureVoices['azure:jenny'];
      const ssml = `<speak version='1.0' xml:lang='en-US'><voice xml:lang='en-US' name='${voiceName}'>${clean.replace(/[<>&'"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;',"'":'&apos;','"':'&quot;'}[c]))}</voice></speak>`;
      const ctrl = new AbortController();
      const tid  = setTimeout(()=>ctrl.abort(), 20000);
      try {
        const r = await fetch(AZURE_TTS_EP, {
          method: 'POST',
          headers: {
            'Ocp-Apim-Subscription-Key': AZURE_TTS_KEY,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-24khz-48kbitrate-mono-mp3',
            'User-Agent': 'ECHOAI/4',
          },
          body: ssml,
          signal: ctrl.signal,
        });
        clearTimeout(tid);
        if(r.status === 401) { AZURE_TTS_BAD = true; log.warn('Azure TTS 401 — key invalid'); return null; }
        if(!r.ok) { const t=await r.text().catch(()=>''); log.warn('Azure TTS '+r.status+': '+t.slice(0,120)); return null; }
        const audio = await r.arrayBuffer();
        if(audio.byteLength < 200) { log.warn('Azure TTS tiny response: '+audio.byteLength+'B'); return null; }
        log.info('Azure TTS OK: '+voiceName+' ('+audio.byteLength+'B)');
        return {audio, provider:'azure', voice:voiceName};
      } catch(e) { clearTimeout(tid); log.warn('Azure TTS error: '+e.message); return null; }
    }

    // Helper: Kokoro TTS via HF — open-source, natural-sounding, free with HF token
    async function tryKokoro(vId) {
      if(!HF_KEY || KOKORO_BAD) return null;
      const voice = (vId && vId.startsWith('kokoro:')) ? vId.slice(7) : 'af_heart';
      const ctrl  = new AbortController();
      const tid   = setTimeout(()=>ctrl.abort(), 35000); // cold start can be slow
      try {
        const r = await fetch(`${HF_EP}/${KOKORO_MODEL}`, {
          method: 'POST',
          headers: {
            'Authorization':    'Bearer '+HF_KEY,
            'Content-Type':     'application/json',
            'x-wait-for-model': 'true',
          },
          body: JSON.stringify({ inputs: clean, parameters: { voice } }),
          signal: ctrl.signal,
        });
        clearTimeout(tid);
        if(r.status === 401) { KOKORO_BAD = true; log.warn('Kokoro/HF 401'); return null; }
        if(!r.ok) {
          const t = await r.text().catch(()=>'');
          log.warn('Kokoro TTS '+r.status+': '+t.slice(0,120));
          // 404 = model not on hf-inference router; disable for session
          if(r.status===404||r.status===410){ KOKORO_BAD=true; log.warn('Kokoro: model not available on HF router — disabling'); }
          return null;
        }
        // HF returns audio/flac or audio/wav — convert via arrayBuffer
        const audio = await r.arrayBuffer();
        if(audio.byteLength < 200) { log.warn('Kokoro TTS tiny: '+audio.byteLength+'B'); return null; }
        const ct = r.headers.get('content-type') || 'audio/flac';
        log.info('Kokoro TTS OK: voice='+voice+' ('+audio.byteLength+'B, '+ct+')');
        return {audio, provider:'kokoro', voice, contentType: ct};
      } catch(e) { clearTimeout(tid); log.warn('Kokoro TTS error: '+e.message); return null; }
    }

    // Route: EL → Edge TTS → Azure → Groq → Kokoro → OpenAI
    let result = null;
    log.info('TTS request: voiceId='+voiceId);
    if(voiceId.startsWith('el:'))       { result = await tryEL(voiceId); }
    else if(voiceId.startsWith('edge:'))   { result = await tryEdgeTTS(voiceId); }
    else if(voiceId.startsWith('groq:'))   { result = await tryGroq(voiceId); }
    else if(voiceId.startsWith('azure:'))  { result = await tryAzure(voiceId); }
    else if(voiceId.startsWith('kokoro:')) { result = await tryKokoro(voiceId); }
    else if(voiceId.startsWith('oai:'))    { result = await tryOAI(voiceId); }

    // Full fallback chain
    if(!result && ELEVENLABS_KEY && !isELBlocked() && !voiceId.startsWith('el:'))
      result = await tryEL('auto');
    if(!result && !EDGE_TTS_BAD)
      result = await tryEdgeTTS('edge:aria');                         // always try — no key needed
    if(!result && AZURE_TTS_KEY && !AZURE_TTS_BAD)
      result = await tryAzure('azure:jenny');
    if(!result && GROQ_KEY && !TTS_BAD.groq)
      result = await tryGroq('auto');
    if(!result && HF_KEY && !KOKORO_BAD)
      result = await tryKokoro('auto');
    if(!result && OPENAI_KEY && !TTS_BAD.openai)
      result = await tryOAI('auto');

    if(!result) {
      const reasons = [];
      if(!ELEVENLABS_KEY)                  reasons.push('EL: no key');
      else if(isELBlocked())               reasons.push('EL: blocked (VPN?)');
      if(!AZURE_TTS_KEY)                   reasons.push('Azure: no key');
      else if(AZURE_TTS_BAD)               reasons.push('Azure: key invalid');
      if(!GROQ_KEY)                        reasons.push('Groq: no key');
      else if(TTS_BAD.groq)                reasons.push('Groq: key invalid');
      if(!HF_KEY)                          reasons.push('Kokoro/HF: no key');
      else if(KOKORO_BAD)                  reasons.push('Kokoro: failed');
      if(!OPENAI_KEY)                      reasons.push('OpenAI: no key');
      else if(TTS_BAD.openai)              reasons.push('OpenAI: key invalid');
      const noKeys = !ELEVENLABS_KEY && !AZURE_TTS_KEY && !GROQ_KEY && !HF_KEY && !OPENAI_KEY;
      const hint = noKeys
        ? 'No TTS keys — add GROQ_API_KEY (free) or AZURE_TTS_KEY (free 500K/mo) to .env'
        : 'All TTS providers failed: '+(reasons.join(' | ')||'check server logs');
      log.warn('TTS all failed: '+hint);
      return res.status(503).json({status:'error', message:hint, code:503, useBrowser:true, reasons});
    }

    const audioContentType = result.contentType || 'audio/mpeg';
    res.setHeader('Content-Type', audioContentType);
    res.setHeader('Content-Length', result.audio.byteLength);
    res.setHeader('X-TTS-Provider', result.provider);
    res.setHeader('X-TTS-Voice', result.voice);
    log.info('TTS: '+result.provider+' / '+result.voice+' ('+result.audio.byteLength+'B)');
    return res.send(Buffer.from(result.audio));

  } catch(e) {
    log.error('TTS route error: '+e.message);
    res.status(502).json(err('TTS error: '+e.message.slice(0,80)));
  }
});


// TTS diagnostic endpoint — returns full EL status without needing server logs
app.get('/api/tts/test', auth, async (req,res) => {
  const diag = { key: !!ELEVENLABS_KEY, fetched: EL_VOICES_FETCHED, voices: EL_VOICE_META.length, tier: EL_TIER||'unknown' };
  if(!ELEVENLABS_KEY) return res.json({ok:false, diag, error:'No ELEVENLABS_API_KEY in .env'});
  if(!EL_VOICES_FETCHED) await fetchELVoices().catch(e=>{ diag.fetchError=e.message; });
  if(!EL_TIER) await fetchELTier().catch(()=>{});
  diag.fetched = EL_VOICES_FETCHED;
  diag.voices  = EL_VOICE_META.length;
  diag.tier    = EL_TIER || 'unknown';
  // Fetch quota/subscription info
  try {
    const sub = await fetch('https://api.elevenlabs.io/v1/user/subscription', { headers:{'xi-api-key':ELEVENLABS_KEY} });
    if(sub.ok) {
      const sd = await sub.json();
      diag.characterCount = sd.character_count;
      diag.characterLimit = sd.character_limit;
      diag.characterPct   = sd.character_limit ? Math.round(sd.character_count/sd.character_limit*100) : '?';
      diag.tier           = sd.tier || sd.billing_tier || EL_TIER || 'unknown';
      diag.nextResetUnix  = sd.next_character_count_reset_unix;
    }
  } catch(e) { diag.subError = e.message; }
  const vid   = (EL_VOICE_META[0]?.id) || Object.values(EL_VOICES)[0] || '21m00Tcm4TlvDq8ikWAM';
  const vname = (EL_VOICE_META[0]?.name) || 'Rachel';
  diag.testVoiceId = vid; diag.testVoiceName = vname;
  const models = ['eleven_flash_v2_5','eleven_turbo_v2_5','eleven_multilingual_v2','eleven_monolingual_v1'];
  for(const model_id of models) {
    try {
      const r = await fetch(`${ELEVENLABS_EP}/${vid}?output_format=mp3_44100_128`, {
        method:'POST',
        headers:{'Content-Type':'application/json','xi-api-key':ELEVENLABS_KEY},
        body: JSON.stringify({ text:'Hello.', model_id, voice_settings:{stability:0.5,similarity_boost:0.75} })
      });
      if(r.ok) {
        const ab = await r.arrayBuffer();
        return res.json({ok:true, diag, model:model_id, bytes:ab.byteLength});
      }
      const errBody = await r.text().catch(()=>'');
      diag['err_'+model_id] = r.status+': '+String(errBody).slice(0,150);
    } catch(e) { diag['err_'+model_id] = e.message; }
  }
  return res.json({ok:false, diag, error:'All models failed — see per-model errors in diag'});
});




// TTS Live Diagnostic — fires a real audio test through each provider
app.get('/api/tts/diagnose', rl, auth, async (req, res) => {
  const testText = 'Hello, I work.';
  const results  = {};

  // Test ElevenLabs
  if(ELEVENLABS_KEY) {
    if(isELBlocked()) {
      results.elevenlabs = { ok:false, error: EL_BLOCKED_REASON || 'Blocked (VPN/unusual_activity)' };
    } else {
      try {
        const vid = Object.values(EL_VOICES)[0] || '21m00Tcm4TlvDq8ikWAM';
        const r = await fetch(`${ELEVENLABS_EP}/${vid}?output_format=mp3_44100_128`, {
          method:'POST',
          headers:{'Content-Type':'application/json','xi-api-key':ELEVENLABS_KEY},
          body: JSON.stringify({text:testText, model_id:'eleven_multilingual_v2', voice_settings:{stability:0.5,similarity_boost:0.75}})
        });
        if(r.ok) {
          const ab = await r.arrayBuffer();
          results.elevenlabs = { ok:true, bytes:ab.byteLength, model:'eleven_multilingual_v2' };
        } else {
          const t = await r.text().catch(()=>'');
          let m=''; try{m=JSON.parse(t)?.detail?.message||t;}catch(_){m=t;}
          results.elevenlabs = { ok:false, status:r.status, error:m.slice(0,120) };
          if(r.status===401||r.status===403) EL_BLOCKED=true;
        }
      } catch(e) { results.elevenlabs = { ok:false, error:e.message.slice(0,100) }; }
    }
  } else { results.elevenlabs = { ok:false, error:'No ELEVENLABS_API_KEY in .env' }; }

  // Test Groq Orpheus
  if(GROQ_KEY) {
    try {
      const r = await fetch('https://api.groq.com/openai/v1/audio/speech', {
        method:'POST',
        headers:{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY},
        body: JSON.stringify({ model:GROQ_TTS_MODEL, input:testText, voice:'troy', response_format:'wav' })
      });
      if(r.ok) {
        const ab = await r.arrayBuffer();
        results.groq = { ok:true, bytes:ab.byteLength, voice:'troy', model:GROQ_TTS_MODEL };
        TTS_BAD.groq = false;
      } else {
        const t = await r.text().catch(()=>'');
        let m=''; try{m=JSON.parse(t)?.error?.message||t;}catch(_){m=t;}
        results.groq = { ok:false, status:r.status, error:m.slice(0,200) };
        if(r.status===401) TTS_BAD.groq = true;
      }
    } catch(e) { results.groq = { ok:false, error:e.message.slice(0,100) }; }
  } else { results.groq = { ok:false, error:'No GROQ_API_KEY in .env' }; }

  // Test OpenAI TTS
  if(OPENAI_KEY) {
    try {
      const r = await fetch('https://api.openai.com/v1/audio/speech', {
        method:'POST',
        headers:{'Content-Type':'application/json','Authorization':'Bearer '+OPENAI_KEY},
        body: JSON.stringify({ model:'tts-1', input:testText, voice:'nova', response_format:'mp3' })
      });
      if(r.ok) {
        const ab = await r.arrayBuffer();
        results.openai = { ok:true, bytes:ab.byteLength };
        TTS_BAD.openai = false;
      } else {
        const t = await r.text().catch(()=>'');
        let m=''; try{m=JSON.parse(t)?.error?.message||t;}catch(_){m=t;}
        results.openai = { ok:false, status:r.status, error:m.slice(0,200) };
        if(r.status===401) TTS_BAD.openai = true;
      }
    } catch(e) { results.openai = { ok:false, error:e.message.slice(0,100) }; }
  } else { results.openai = { ok:false, error:'No OPENAI_API_KEY in .env' }; }

  const anyOk = Object.values(results).some(r=>r.ok);
  log.info('TTS diagnose: '+JSON.stringify(results));
  res.json({ status:'success', data:{ results, anyOk,
    recommendation: anyOk
      ? 'TTS working — active provider: '+(results.groq?.ok?'Groq Orpheus':results.elevenlabs?.ok?'ElevenLabs':'OpenAI')
      : 'All TTS providers failed — check your .env API keys and server logs above for exact errors'
  }});
});

// Vision provider status
app.get('/api/vision/status', auth, (req,res) => {
  res.json({status:'success', data:{
    gemini:  { configured:!!GEMINI_KEY, bad:BAD_KEYS.gemini },
    openai:  { configured:!!OPENAI_KEY, bad:BAD_KEYS.openai },
    hasVision: !!(( GEMINI_KEY&&!BAD_KEYS.gemini) || (OPENAI_KEY&&!BAD_KEYS.openai)),
    activeProvider: (GEMINI_KEY&&!BAD_KEYS.gemini) ? 'Gemini'
      : (OPENAI_KEY&&!BAD_KEYS.openai) ? 'OpenAI' : 'none',
  }});
});

// Reset ElevenLabs block (manual recovery)
app.post('/api/tts/reset', rl, auth, (req,res) => {
  EL_BLOCKED     = false;
  EL_NET_ERROR_AT = 0;
  EL_BLOCKED_REASON = '';
  EL_VOICES_FETCHED = false;
  log.info('EL block manually reset by '+req.uname);
  // Re-fetch voices in background
  if(ELEVENLABS_KEY) fetchELVoices().catch(()=>{});
  res.json({status:'success', message:'ElevenLabs block cleared — re-fetching voices'});
});

// TTS provider status
app.get('/api/tts/status', auth, (req,res) => {
  res.json({status:'success', data:{
    elevenlabs: { configured:!!ELEVENLABS_KEY, blocked:EL_BLOCKED, softBlocked:!!(EL_NET_ERROR_AT&&Date.now()-EL_NET_ERROR_AT<600000), reason:EL_BLOCKED_REASON, voices:EL_VOICE_META.length, tier:EL_TIER||'unknown' },
    edge:       { configured:true, bad:EDGE_TTS_BAD },
    groq:       { configured:!!GROQ_KEY,       bad:TTS_BAD.groq },
    azure:      { configured:!!AZURE_TTS_KEY,  bad:AZURE_TTS_BAD },
    kokoro:     { configured:!!HF_KEY,         bad:KOKORO_BAD },
    openai:     { configured:!!OPENAI_KEY,     bad:TTS_BAD.openai },
    active: (ELEVENLABS_KEY&&!isELBlocked()) ? 'elevenlabs'
      : !EDGE_TTS_BAD ? 'edge'
      : (AZURE_TTS_KEY&&!AZURE_TTS_BAD) ? 'azure'
      : (GROQ_KEY&&!TTS_BAD.groq) ? 'groq'
      : (HF_KEY&&!KOKORO_BAD) ? 'kokoro'
      : (OPENAI_KEY&&!TTS_BAD.openai) ? 'openai' : 'browser',
  }});
});

// Debug route — shows provider status (no sensitive data)
app.get('/api/debug', auth, (req,res) => {
  res.json({
    status:'success',
    data:{
      providers:{
        groq:    { configured:!!GROQ_KEY,    keyPrefix:GROQ_KEY?GROQ_KEY.slice(0,8)+'...':'none', bad:BAD_KEYS.groq,  model:GROQ_MODEL },
        gemini:  { configured:!!GEMINI_KEY,  keyPrefix:GEMINI_KEY?GEMINI_KEY.slice(0,8)+'...':'none', bad:BAD_KEYS.gemini, model:GEMINI_MODEL },
        openai:  { configured:!!OPENAI_KEY,  keyPrefix:OPENAI_KEY?OPENAI_KEY.slice(0,8)+'...':'none', bad:BAD_KEYS.openai, model:OPENAI_MODEL },
        elevenlabs:{ configured:!!ELEVENLABS_KEY, keyPrefix:ELEVENLABS_KEY?ELEVENLABS_KEY.slice(0,8)+'...':'none' },
      },
      queue:{ dead:OAQ.DEAD, queueLen:OAQ.queue.length, cooldown:OAQ.cooldown },
      uptime: process.uptime()+'s',
    }
  });
});

// Reset bad key flags manually
app.post('/api/debug/reset', rl, auth, (req,res) => {
  BAD_KEYS.groq=false; BAD_KEYS.gemini=false; BAD_KEYS.openai=false;
  OAQ.DEAD=false; OAQ.fails=0; OAQ.cooldown=0;
  log.info('Debug reset triggered by '+req.uname);
  res.json({status:'success', message:'All provider flags reset'});
});

// 404 fallback
app.use((_req,res)=>res.status(404).json(err('Not found',404)));

// =============================================================================
// START
// =============================================================================
const d=rd();
const nAcc=Object.keys(d.accounts||{}).length;

server.listen(PORT,()=>{
  console.log(`
  ╔══════════════════════════════════════════════╗
  ║            ECHOAI v4.0 — RUNNING            ║
  ╠══════════════════════════════════════════════╣
  ● Port      : ${PORT}
  ● Provider  : ${AI_PROVIDER}
  ● Model     : ${ACTIVE_MODEL}
  ● Vision    : ${GEMINI_KEY?'✓ Gemini 2.0-flash (FREE 15 RPM)':OPENAI_KEY?'✓ OpenAI gpt-4o-mini':'✗ None — add GEMINI_API_KEY (free at aistudio.google.com)'}
  ● Gemini    : ${GEMINI_KEY?'✓ FREE 15 RPM (vision fallback)':'✗ None — aistudio.google.com (free)'}
  ● Groq      : ${GROQ_KEY?'✓ FREE 30 RPM':'✗ None — console.groq.com (free)'}
  ● OpenAI    : ${OPENAI_KEY?'✓ Loaded (paid fallback)':'✗ None'}
  ● Users     : ${nAcc}
  ● TTS       : ${ELEVENLABS_KEY?'✓ ElevenLabs (studio quality)':'✓ Edge TTS (FREE — Azure Neural, no key needed)'}${GROQ_KEY?' + Groq Orpheus':''}${AZURE_TTS_KEY?' + Azure':''}
  ╠══════════════════════════════════════════════╣
  ● Start tunnel in NEW terminal:
    ssh -R 80:localhost:${PORT} nokey@localhost.run
  ╚══════════════════════════════════════════════╝
  `);
});

// Fetch real ElevenLabs voices on startup
if(ELEVENLABS_KEY) setTimeout(()=>{ fetchELVoices().catch(()=>{}); fetchELTier().catch(()=>{}); }, 2000);

process.on('uncaughtException', e=>log.error('Uncaught',{e:e.message}));
process.on('unhandledRejection', e=>log.error('Unhandled',{e:String(e)}));
