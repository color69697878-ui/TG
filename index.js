import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import express from 'express';
import dotenv from 'dotenv';
import { Telegraf } from 'telegraf';
import OpenAI from 'openai';

dotenv.config();

const {
  TELEGRAM_BOT_TOKEN,
  OPENAI_API_KEY,
  OWNER_TELEGRAM_ID,
  OPENAI_MODEL = 'gpt-4.1-mini',
  PORT = 3000,
  RENDER_EXTERNAL_URL,
  WEBHOOK_PATH = '/telegram/webhook',
  TELEGRAM_WEBHOOK_SECRET,
  ALLOWED_CHAT_IDS = ''
} = process.env;

if (!TELEGRAM_BOT_TOKEN) throw new Error('缺少 TELEGRAM_BOT_TOKEN');
if (!OPENAI_API_KEY) throw new Error('缺少 OPENAI_API_KEY');

const app = express();
const bot = new Telegraf(TELEGRAM_BOT_TOKEN);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), 'data');
const SETTINGS_FILE = path.join(DATA_DIR, 'chatSettings.json');
const AUTH_FILE = path.join(DATA_DIR, 'allowedChats.json');

ensureDir(DATA_DIR);
const settingsStore = loadJson(SETTINGS_FILE, {});
const authStore = loadJson(AUTH_FILE, {
  allowedChats: parseAllowedChatIds(ALLOWED_CHAT_IDS)
});

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function loadJson(file, fallback) {
  try {
    if (!fs.existsSync(file)) {
      fs.writeFileSync(file, JSON.stringify(fallback, null, 2), 'utf8');
      return structuredCloneSafe(fallback);
    }
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    console.error(`讀取 ${file} 失敗，改用預設值：`, error.message);
    return structuredCloneSafe(fallback);
  }
}

function saveJson(file, value) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2), 'utf8');
}

function structuredCloneSafe(value) {
  return JSON.parse(JSON.stringify(value));
}

function parseAllowedChatIds(raw) {
  return String(raw || '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
}

function getChatId(ctx) {
  return String(ctx.chat?.id || '');
}

function isOwner(ctx) {
  return String(ctx.from?.id || '') === String(OWNER_TELEGRAM_ID || '');
}

function isGroup(ctx) {
  return ['group', 'supergroup'].includes(ctx.chat?.type);
}

function isAllowedChat(chatId) {
  const allowList = new Set((authStore.allowedChats || []).map(String));
  if (allowList.size === 0) return true; // 空白代表全部允許
  return allowList.has(String(chatId));
}

function allowChat(chatId) {
  authStore.allowedChats = Array.from(new Set([...(authStore.allowedChats || []).map(String), String(chatId)]));
  saveJson(AUTH_FILE, authStore);
}

function disallowChat(chatId) {
  authStore.allowedChats = (authStore.allowedChats || []).map(String).filter((id) => id !== String(chatId));
  saveJson(AUTH_FILE, authStore);
}

function getDefaultSettings() {
  return {
    enabled: true,
    mode: 'auto', // auto | zh2th | th2zh | manual
    showOriginal: false,
    target: 'auto', // auto | zh | th | en
    ignoreCommands: true,
    ignorePureEmoji: true,
    ignoreShortSymbols: true
  };
}

function getChatSettings(chatId) {
  if (!settingsStore[chatId]) {
    settingsStore[chatId] = getDefaultSettings();
    saveJson(SETTINGS_FILE, settingsStore);
  }
  return settingsStore[chatId];
}

function updateChatSettings(chatId, patch) {
  const current = getChatSettings(chatId);
  settingsStore[chatId] = { ...current, ...patch };
  saveJson(SETTINGS_FILE, settingsStore);
  return settingsStore[chatId];
}

function normalizeText(text) {
  return String(text || '').replace(/\r/g, '').trim();
}

function isOnlySymbolsOrWhitespace(text) {
  const t = normalizeText(text);
  if (!t) return true;
  return /^[\p{P}\p{S}\s]+$/u.test(t);
}

function isOnlyEmoji(text) {
  const t = normalizeText(text);
  if (!t) return false;
  const withoutEmoji = t.replace(/[\p{Extended_Pictographic}\uFE0F\u200D]/gu, '').replace(/\s/g, '');
  return withoutEmoji.length === 0;
}

function containsChinese(text) {
  return /[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/u.test(text);
}

function containsThai(text) {
  return /[\u0E00-\u0E7F]/u.test(text);
}

function detectDirection(text, mode, target) {
  const hasZh = containsChinese(text);
  const hasTh = containsThai(text);

  if (mode === 'zh2th') return { source: 'zh', target: 'th' };
  if (mode === 'th2zh') return { source: 'th', target: 'zh' };
  if (mode === 'manual') {
    if (target && target !== 'auto') {
      const source = hasZh ? 'zh' : hasTh ? 'th' : 'auto';
      return { source, target };
    }
    return null;
  }

  if (hasZh && !hasTh) return { source: 'zh', target: 'th' };
  if (hasTh && !hasZh) return { source: 'th', target: 'zh' };
  if (hasZh && hasTh) return { source: 'mixed', target: 'zh' };

  if (target && target !== 'auto') return { source: 'auto', target };
  return null;
}

function shouldIgnoreMessage(ctx, text, settings) {
  if (!settings.enabled) return true;
  if (!text) return true;
  if (ctx.message?.via_bot) return true;
  if (ctx.from?.is_bot) return true;
  if (settings.ignoreCommands && text.startsWith('/')) return true;
  if (settings.ignoreShortSymbols && isOnlySymbolsOrWhitespace(text)) return true;
  if (settings.ignorePureEmoji && isOnlyEmoji(text)) return true;
  return false;
}

function buildSystemPrompt(direction, showOriginal) {
  const mapping = {
    zh: '繁體中文',
    th: '泰文',
    en: '英文',
    auto: '自動判斷語言'
  };

  return [
    '你是專業聊天翻譯助手。',
    '任務是把使用者訊息翻譯成自然、口語、符合聊天情境的目標語言。',
    '保留原本語氣、曖昧感、禮貌程度、數字、時間、地點、品牌名、emoji、換行。',
    '不要解釋，不要加前言，不要加括號說明，不要審稿。',
    '若原文已經是目標語言，請輸出更自然但意思不變的版本。',
    '遇到只有代號、IN/OUT、房號、日期、金額、電話、代稱時，盡量保留原樣。',
    `來源語言：${mapping[direction?.source || 'auto']}`,
    `目標語言：${mapping[direction?.target || 'auto']}`,
    showOriginal
      ? '輸出格式必須是：原文：<原文>\n翻譯：<翻譯>'
      : '只輸出翻譯結果本身。'
  ].join('\n');
}

async function translateText(text, direction, showOriginal) {
  const input = normalizeText(text);
  if (!input) return '';

  const response = await openai.responses.create({
    model: OPENAI_MODEL,
    input: [
      {
        role: 'system',
        content: [{ type: 'input_text', text: buildSystemPrompt(direction, showOriginal) }]
      },
      {
        role: 'user',
        content: [{ type: 'input_text', text: input }]
      }
    ]
  });

  const output = response.output_text?.trim?.() || '';
  if (!output) throw new Error('OpenAI 沒有回傳翻譯內容');
  return output;
}

async function safeReply(ctx, text, extra = {}) {
  if (!text) return;
  try {
    await ctx.reply(text, {
      disable_web_page_preview: true,
      ...extra
    });
  } catch (error) {
    console.error('回覆失敗：', error.message);
  }
}

bot.catch((error, ctx) => {
  console.error('Bot error:', error);
  if (ctx?.chat?.id) {
    safeReply(ctx, '抱歉，剛剛翻譯時發生錯誤，請再試一次。');
  }
});

bot.start(async (ctx) => {
  const chatId = getChatId(ctx);
  const settings = getChatSettings(chatId);
  await safeReply(
    ctx,
    [
      'Telegram 中泰翻譯機器人已啟動 ✅',
      '',
      '常用指令：',
      '/help 查看完整教學',
      '/mode auto 自動中泰雙向翻譯',
      '/mode zh2th 固定中翻泰',
      '/mode th2zh 固定泰翻中',
      '/showoriginal on|off 顯示/隱藏原文',
      '/status 查看目前設定',
      '/translate 文字 手動翻譯一次',
      '',
      `目前模式：${settings.mode}`
    ].join('\n')
  );
});

bot.help(async (ctx) => {
  await safeReply(
    ctx,
    [
      '【Telegram 翻譯機器人指令】',
      '/start 啟動',
      '/help 說明',
      '/myid 查看你的 Telegram ID',
      '/chatid 查看目前聊天室 ID',
      '/status 查看此聊天室設定',
      '/on 開啟自動翻譯',
      '/off 關閉自動翻譯',
      '/mode auto 自動中泰雙向翻譯',
      '/mode zh2th 固定中翻泰',
      '/mode th2zh 固定泰翻中',
      '/mode manual 手動模式',
      '/settarget zh|th|en 設定手動模式目標語言',
      '/showoriginal on|off 是否顯示原文',
      '/translate 文字 只翻這一句',
      '/authorize 允許目前聊天室（限擁有者）',
      '/deauthorize 移除此聊天室（限擁有者）'
    ].join('\n')
  );
});

bot.command('myid', async (ctx) => {
  await safeReply(ctx, `你的 Telegram ID：${ctx.from?.id || '未知'}`);
});

bot.command('chatid', async (ctx) => {
  await safeReply(ctx, `目前聊天室 ID：${ctx.chat?.id || '未知'}`);
});

bot.command('status', async (ctx) => {
  const chatId = getChatId(ctx);
  const s = getChatSettings(chatId);
  const allowed = isAllowedChat(chatId) ? '是' : '否';
  await safeReply(
    ctx,
    [
      `聊天室：${ctx.chat?.title || ctx.from?.first_name || '未命名'}`,
      `chat_id：${chatId}`,
      `白名單允許：${allowed}`,
      `啟用狀態：${s.enabled ? '開啟' : '關閉'}`,
      `翻譯模式：${s.mode}`,
      `手動目標語言：${s.target}`,
      `顯示原文：${s.showOriginal ? '開啟' : '關閉'}`
    ].join('\n')
  );
});

bot.command('on', async (ctx) => {
  const chatId = getChatId(ctx);
  updateChatSettings(chatId, { enabled: true });
  await safeReply(ctx, '已開啟這個聊天室的自動翻譯。');
});

bot.command('off', async (ctx) => {
  const chatId = getChatId(ctx);
  updateChatSettings(chatId, { enabled: false });
  await safeReply(ctx, '已關閉這個聊天室的自動翻譯。');
});

bot.command('showoriginal', async (ctx) => {
  const text = normalizeText(ctx.message?.text || '');
  const value = text.split(/\s+/)[1]?.toLowerCase();
  if (!['on', 'off'].includes(value)) {
    return safeReply(ctx, '用法：/showoriginal on 或 /showoriginal off');
  }
  const chatId = getChatId(ctx);
  updateChatSettings(chatId, { showOriginal: value === 'on' });
  await safeReply(ctx, `顯示原文已${value === 'on' ? '開啟' : '關閉'}。`);
});

bot.command('settarget', async (ctx) => {
  const text = normalizeText(ctx.message?.text || '');
  const value = text.split(/\s+/)[1]?.toLowerCase();
  if (!['zh', 'th', 'en', 'auto'].includes(value)) {
    return safeReply(ctx, '用法：/settarget zh 或 /settarget th 或 /settarget en 或 /settarget auto');
  }
  const chatId = getChatId(ctx);
  updateChatSettings(chatId, { target: value });
  await safeReply(ctx, `手動模式目標語言已設定為：${value}`);
});

bot.command('mode', async (ctx) => {
  const text = normalizeText(ctx.message?.text || '');
  const value = text.split(/\s+/)[1]?.toLowerCase();
  if (!['auto', 'zh2th', 'th2zh', 'manual'].includes(value)) {
    return safeReply(ctx, '用法：/mode auto 或 /mode zh2th 或 /mode th2zh 或 /mode manual');
  }
  const chatId = getChatId(ctx);
  updateChatSettings(chatId, { mode: value, enabled: true });
  const tips = {
    auto: '自動中泰雙向翻譯',
    zh2th: '固定中翻泰',
    th2zh: '固定泰翻中',
    manual: '手動模式，請搭配 /settarget 使用'
  };
  await safeReply(ctx, `翻譯模式已切換為：${value}\n${tips[value]}`);
});

bot.command('authorize', async (ctx) => {
  if (!isOwner(ctx)) return safeReply(ctx, '只有擁有者可以授權聊天室。');
  const chatId = getChatId(ctx);
  allowChat(chatId);
  await safeReply(ctx, `已授權聊天室：${chatId}`);
});

bot.command('deauthorize', async (ctx) => {
  if (!isOwner(ctx)) return safeReply(ctx, '只有擁有者可以取消授權聊天室。');
  const chatId = getChatId(ctx);
  disallowChat(chatId);
  await safeReply(ctx, `已取消授權聊天室：${chatId}`);
});

bot.command('translate', async (ctx) => {
  const raw = normalizeText(ctx.message?.text || '');
  const content = raw.replace(/^\/translate(@\w+)?\s*/i, '');
  if (!content) return safeReply(ctx, '用法：/translate 你要翻譯的文字');

  const chatId = getChatId(ctx);
  if (!isAllowedChat(chatId)) return;

  const settings = getChatSettings(chatId);
  const direction = detectDirection(content, settings.mode, settings.target) || { source: 'auto', target: 'zh' };
  const translated = await translateText(content, direction, settings.showOriginal);
  await safeReply(ctx, translated, {
    reply_parameters: { message_id: ctx.message.message_id }
  });
});

bot.on('text', async (ctx) => {
  const chatId = getChatId(ctx);
  if (!chatId) return;
  if (!isAllowedChat(chatId)) return;

  const text = normalizeText(ctx.message?.text || '');
  const settings = getChatSettings(chatId);
  if (shouldIgnoreMessage(ctx, text, settings)) return;

  // 群組中若是白名單模式，避免機器人太吵：
  // 只要已啟用，就自動翻；私人聊天也自動翻。
  const direction = detectDirection(text, settings.mode, settings.target);
  if (!direction) return;

  const translated = await translateText(text, direction, settings.showOriginal);

  if (!translated || normalizeText(translated) === normalizeText(text)) return;

  await safeReply(ctx, translated, {
    reply_parameters: { message_id: ctx.message.message_id }
  });
});

app.get('/', (_req, res) => {
  res.status(200).send('Telegram Translate Bot is running.');
});

app.post(WEBHOOK_PATH, express.json({ limit: '2mb' }), async (req, res) => {
  try {
    if (TELEGRAM_WEBHOOK_SECRET) {
      const received = req.headers['x-telegram-bot-api-secret-token'];
      if (received !== TELEGRAM_WEBHOOK_SECRET) {
        return res.status(401).send('Invalid secret token');
      }
    }
    await bot.handleUpdate(req.body, res);
    if (!res.headersSent) res.sendStatus(200);
  } catch (error) {
    console.error('Webhook error:', error);
    if (!res.headersSent) res.sendStatus(500);
  }
});

async function setupWebhook() {
  if (!RENDER_EXTERNAL_URL) {
    console.log('未設定 RENDER_EXTERNAL_URL，略過自動 webhook 設定。');
    return;
  }

  const cleanBase = RENDER_EXTERNAL_URL.replace(/\/$/, '');
  const fullWebhookUrl = `${cleanBase}${WEBHOOK_PATH}`;

  try {
    await bot.telegram.setWebhook(fullWebhookUrl, {
      secret_token: TELEGRAM_WEBHOOK_SECRET || crypto.randomBytes(24).toString('hex')
    });
    const me = await bot.telegram.getMe();
    console.log(`Webhook 已設定：${fullWebhookUrl}`);
    console.log(`Bot username：@${me.username}`);
  } catch (error) {
    console.error('設定 webhook 失敗：', error.message);
  }
}

app.listen(Number(PORT), async () => {
  console.log(`Server running on port ${PORT}`);
  await setupWebhook();
});

process.once('SIGINT', () => process.exit(0));
process.once('SIGTERM', () => process.exit(0));
