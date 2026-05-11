import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const WORKFLOW_TEMPLATES_BASE = "/workflow_templates/ComfyUI-SparkVSR";
const LATEST_WORKFLOW_API = "/sparkvsr/latest_workflow";
const FALLBACK_WORKFLOW = "sparkvsr_all_modes_preview.json";
const VERSION_STORAGE_KEY = "SparkVSR.LoadedWorkflowVersion";
const BOOT_WINDOW_MS = 8000;

const STOCK_DEFAULT_NODE_TYPES = new Set([
  "CheckpointLoaderSimple",
  "CLIPTextEncode",
  "EmptyLatentImage",
  "KSampler",
  "VAEDecode",
  "SaveImage",
]);

let bootDeadline = 0;
let loadingSparkVSRWorkflow = false;

function getGraphNodes() {
  return (app.graph && app.graph._nodes) || [];
}

function hasSparkVSRNodes(nodes) {
  return nodes.some((node) => node && node.type && node.type.startsWith("SparkVSR"));
}

function isStockDefaultGraph(nodes) {
  if (nodes.length === 0) return true;
  if (nodes.length > 8) return false;
  return nodes.every((node) => node && STOCK_DEFAULT_NODE_TYPES.has(node.type));
}

function isAutoLoadEnabled() {
  try {
    const v = app.ui.settings.getSettingValue("SparkVSR.AutoLoadDefaultWorkflow");
    return v !== false;
  } catch (_) {
    return true;
  }
}

async function fetchLatestWorkflow() {
  try {
    const res = await fetch(api.apiURL(LATEST_WORKFLOW_API));
    if (res.ok) {
      const data = await res.json();
      if (data.filename) {
        const url = `${WORKFLOW_TEMPLATES_BASE}/${data.filename}`;
        const wfRes = await fetch(api.apiURL(url));
        if (wfRes.ok) {
          const workflow = await wfRes.json();
          return { url, workflow };
        }
      }
    }
  } catch (_) {}

  // Fallback
  const url = `${WORKFLOW_TEMPLATES_BASE}/${FALLBACK_WORKFLOW}`;
  try {
    const res = await fetch(api.apiURL(url));
    if (res.ok) return { url, workflow: await res.json() };
  } catch (_) {}
  return null;
}

async function tryLoadWorkflow() {
  if (loadingSparkVSRWorkflow || Date.now() > bootDeadline) return;
  if (!isAutoLoadEnabled()) return;

  const result = await fetchLatestWorkflow();
  if (!result) return;

  const { url, workflow } = result;
  const latestVersion = String(workflow._sparkvsr_version || "1");
  const storedVersion = localStorage.getItem(VERSION_STORAGE_KEY) || "";

  const nodes = getGraphNodes();
  const hasSparkVSR = hasSparkVSRNodes(nodes);
  const isBlankOrStock = isStockDefaultGraph(nodes);

  // Load when: blank/stock canvas, OR version changed (forces widget realignment)
  const shouldLoad = isBlankOrStock || !hasSparkVSR || latestVersion !== storedVersion;
  if (!shouldLoad) return;

  loadingSparkVSRWorkflow = true;
  try {
    await app.loadGraphData(workflow);
    if (app.graph && app.graph.setDirtyCanvas) app.graph.setDirtyCanvas(true, true);
    localStorage.setItem(VERSION_STORAGE_KEY, latestVersion);
    console.info(`[SparkVSR] Loaded workflow v${latestVersion}: ${url}`);
  } catch (err) {
    console.warn("[SparkVSR] Could not load default workflow:", err);
  } finally {
    loadingSparkVSRWorkflow = false;
  }
}

function schedule(delayMs = 0) {
  window.setTimeout(tryLoadWorkflow, delayMs);
}

app.registerExtension({
  name: "SparkVSR.DefaultWorkflow",
  settings: [
    {
      id: "SparkVSR.AutoLoadDefaultWorkflow",
      category: ["SparkVSR", "Workflow", "Auto-load Default Workflow"],
      name: "Auto-load SparkVSR workflow on startup",
      tooltip:
        "When ComfyUI opens on a blank canvas or when a new workflow version is available, automatically load the SparkVSR default workflow.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async setup() {
    bootDeadline = Date.now() + BOOT_WINDOW_MS;

    const origLoad = app.loadGraphData && app.loadGraphData.bind(app);
    if (origLoad) {
      app.loadGraphData = async function (...args) {
        const result = await origLoad(...args);
        schedule();
        return result;
      };
    }

    schedule(500);
    schedule(1500);
    schedule(3000);
  },
});
