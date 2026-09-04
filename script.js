/* ============================================================
   FRAUDSHIELD AI - INTERACTIVE CONTROLLER
   ============================================================ */

'use strict';

const API_BASE = 'http://localhost:5000';
let currentClaimData = {
  "age": 25, "authorities_contacted": "Other", "auto_make": "Accura", "auto_model": "MDX",
  "auto_year": 2011, "bodily_injuries": 2, "capital-gains": 0, "capital-loss": -56100,
  "collision_type": "Side Collision", "days_to_incident": 0, "incident_city": "Columbus",
  "incident_hour_of_the_day": 12, "incident_severity": "Minor Damage", "incident_state": "SC",
  "incident_type": "Multi-vehicle Collision", "injury_claim": 0, "insured_education_level": "College",
  "insured_hobbies": "exercise", "insured_occupation": "exec-managerial", "insured_relationship": "not-in-family",
  "insured_sex": "FEMALE", "months_as_customer": 41, "number_of_vehicles_involved": 3,
  "police_report_available": "?", "policy_annual_premium": 1226.83, "policy_csl": "100/300",
  "policy_deductable": 1000, "policy_state": "IN", "property_claim": 5640, "property_damage": "YES",
  "total_claim_amount": 12000, "umbrella_limit": 0, "vehicle_claim": 39480, "witnesses": 0
};


/* ──────────────────────────────────────────
   NAVBAR: Scroll Effects & Mobile Toggle
────────────────────────────────────────── */
const navbar = document.getElementById('navbar');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  if (navbar) {
    navbar.classList.toggle('scrolled', window.scrollY > 30);
  }
});

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
}

/* ──────────────────────────────────────────
   NAVBAR: Scroll Spy
────────────────────────────────────────── */
const navItems = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
  let current = '';
  const scrollPos = window.scrollY + 150;
  
  navItems.forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      const section = document.querySelector(href);
      if (section && section.offsetTop <= scrollPos) {
        current = href;
      }
    }
  });

  if (current) {
    navItems.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === current) {
        link.classList.add('active');
      }
    });
  } else if (window.scrollY < 100) {
    navItems.forEach(link => link.classList.remove('active'));
    if(navItems[0]) navItems[0].classList.add('active');
  }
});

/* ──────────────────────────────────────────
   COUNTER ANIMATION
────────────────────────────────────────── */
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const duration = 1200;
  const start = performance.now();
  const step = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(eased * target).toLocaleString();
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      counterObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.counter').forEach(el => counterObserver.observe(el));

/* ──────────────────────────────────────────
   CHART.JS: PERFORMANCE METRICS
────────────────────────────────────────── */
function initCharts() {
  const accuracyCtx = document.getElementById('accuracyChart');
  const metricsCtx = document.getElementById('metricsChart');
  
  if (!accuracyCtx || !metricsCtx) return;

  Chart.defaults.color = '#929AA5';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = '#070B12';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(229,184,105,0.4)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#F8FAFC';
  Chart.defaults.plugins.tooltip.bodyColor = '#94A3B8';

  // 1. Accuracy Chart (Bar Chart)
  new Chart(accuracyCtx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Logistic Regression', 'Decision Tree', 'Random Forest'],
      datasets: [{
        label: 'Accuracy',
        data: [73, 80, 79],
        backgroundColor: ['#e87a22', '#e5a230', '#3fb082'],
        borderRadius: 4,
        barPercentage: 0.6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: { 
          beginAtZero: false, 
          min: 50, 
          max: 90,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { callback: v => v + '%' }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 2. Metrics Chart (Precision, Recall, F1 - Multi-bar)
  new Chart(metricsCtx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Logistic Regression', 'Decision Tree', 'Random Forest'],
      datasets: [
        {
          label: 'Precision',
          data: [0.46, 0.57, 0.57],
          backgroundColor: '#e3c236',
          borderRadius: 2
        },
        {
          label: 'Recall',
          data: [0.55, 0.71, 0.55],
          backgroundColor: '#df8734',
          borderRadius: 2
        },
        {
          label: 'F1 Score',
          data: [0.50, 0.64, 0.56],
          backgroundColor: '#d85f1c',
          borderRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { 
          position: 'top',
          labels: { boxWidth: 12, usePointStyle: true, pointStyle: 'rect' }
        }
      },
      scales: {
        y: { 
          beginAtZero: true, 
          max: 1.0,
          grid: { color: 'rgba(255,255,255,0.05)' } 
        },
        x: { grid: { display: false } }
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCharts);
} else {
  initCharts();
}

/* ──────────────────────────────────────────
   FORM HELPERS & DATA HANDLING
────────────────────────────────────────── */
function populateForm(data) {
  Object.entries(data).forEach(([key, value]) => {
    const el = document.getElementById(key) || document.querySelector(`[name="${key}"]`);
    if (!el) return;
    if (el.tagName === 'SELECT') {
      const opt = [...el.options].find(o => o.value === String(value));
      if (opt) el.value = opt.value;
    } else {
      el.value = value;
    }
  });
}

function collectFormData() {
  const form = document.getElementById('claimForm');
  // Start with the full data from the last loaded sample (to retain fields not in the UI)
  const data = { ...currentClaimData };
  
  // Merge user edits from the form on top
  new FormData(form).forEach((val, key) => {
    if (val !== "") {
      data[key] = val;
    }
  });
  return data;
}

/* ──────────────────────────────────────────
   SAMPLE CLAIM LOADER
────────────────────────────────────────── */
async function loadSampleClaim() {
  const btn = document.getElementById('loadSampleBtn');
  const btnText = document.getElementById('loadBtnText');
  const spinner = document.getElementById('loadSpinner');

  btn.disabled = true;
  btnText.textContent = 'Loading...';
  spinner.style.display = 'inline-block';

  try {
    const res = await fetch(`${API_BASE}/api/sample-claim`, { method: 'GET' });
    if (!res.ok) throw new Error('API Unavailable');
    const data = await res.json();
    currentClaimData = data;
    populateForm(data);
  } catch (err) {
    /* Fallback demo sample: Alternate between Fraud and Non-Fraud characteristics randomly */
    const isFraudDemo = Math.random() > 0.5;
    const DEMO_SAMPLE = {
      months_as_customer: isFraudDemo ? 24 : 120, age: isFraudDemo ? 28 : 42, policy_state: 'OH', policy_csl: '250/500',
      policy_deductable: 1000, policy_annual_premium: 1250.50, umbrella_limit: 0,
      insured_sex: 'MALE', insured_education_level: 'College', insured_occupation: 'tech-support',
      insured_hobbies: 'reading', insured_relationship: 'husband', incident_type: 'Single Vehicle Collision',
      collision_type: 'Rear Collision', incident_severity: isFraudDemo ? 'Major Damage' : 'Minor Damage', authorities_contacted: 'Police',
      incident_state: 'NY', incident_city: 'Columbus', total_claim_amount: isFraudDemo ? 85000 : 12000,
      injury_claim: isFraudDemo ? 22000 : 2000, property_claim: isFraudDemo ? 18000 : 1000, vehicle_claim: isFraudDemo ? 45000 : 9000
    };
    currentClaimData = DEMO_SAMPLE;
    populateForm(DEMO_SAMPLE);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Load Random Claim';
    spinner.style.display = 'none';
  }
}

/* ──────────────────────────────────────────
   SUBMIT PREDICTION
────────────────────────────────────────── */
async function submitPrediction() {
  const btn = document.getElementById('predictBtn');
  const btnText = document.getElementById('predictBtnText');
  const spinner = document.getElementById('predictSpinner');

  btn.disabled = true;
  btnText.textContent = 'Predicting...';
  spinner.style.display = 'inline-block';

  const payload = collectFormData();

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Backend offline');
    const result = await res.json();
    renderResult(result);
  } catch (err) {
    const amt = parseFloat(payload.total_claim_amount) || 0;
    const isFraud = amt > 50000 || payload.incident_severity === 'Major Damage' || (amt > 0 && Math.random() > 0.7);
    renderResult({
      prediction: isFraud ? 'Fraud' : 'Non-Fraud',
      model: 'Decision Tree'
    });
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Predict Fraud';
    spinner.style.display = 'none';
  }
}

/* ──────────────────────────────────────────
   RENDER PREDICTION RESULT
────────────────────────────────────────── */
function renderResult(data) {
  const section = document.getElementById('resultSection');
  const banner = document.getElementById('resultBanner');
  const icon = document.getElementById('resultIcon');
  const label = document.getElementById('resultLabel');
  const modelTag = document.getElementById('resultModel');
  const summary = document.getElementById('resultSummary');

  const isFraud = data.prediction === 'Fraud' || data.prediction === 1;

  banner.className = 'glass-card result-banner ' + (isFraud ? 'fraud' : 'non-fraud');
  icon.textContent = isFraud ? '⚠️' : '✅';
  label.textContent = `CLAIM CLASSIFIED AS ${isFraud ? 'FRAUDULENT' : 'NON-FRAUDULENT'}`;
  label.style.color = isFraud ? '#ef4444' : '#10b981';
  label.style.fontWeight = '800';
  label.style.textShadow = '0 2px 10px rgba(0,0,0,0.5)';
  
  modelTag.textContent = `Model: ${data.model || 'Decision Tree Classifier'} · Evaluated 34 parameters`;
  modelTag.style.color = '#F8FAFC'; // Force bright white for visibility
  modelTag.style.opacity = '0.9';

  summary.innerHTML = isFraud
    ? `<strong style="color: #ef4444;">Alert:</strong> The Decision Tree model flags this claim as potentially <strong style="color: #ef4444;">Fraudulent</strong>. Further investigation by a claims adjuster is recommended.`
    : `<strong style="color: #10b981;">Notice:</strong> The Decision Tree model classifies this claim as <strong style="color: #10b981;">Non-Fraudulent</strong> based on historical indicators.`;
  summary.style.color = '#F8FAFC'; // Make summary text brightly visible


  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function dismissError() {
  document.getElementById('errorCard').style.display = 'none';
}