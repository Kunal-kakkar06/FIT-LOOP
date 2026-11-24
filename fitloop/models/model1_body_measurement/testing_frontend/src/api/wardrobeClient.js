/**
 * api/wardrobeClient.js
 * --------------------
 * Client methods for Wardrobe ML (garment detection, color extraction,
 * duplicate prevention, and integrated fit passport).
 */

// Use relative path so Vite proxy routes to backend directly without CORS/host restrictions
const API_BASE = '';

export async function fetchSystemHealth() {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function getOrCreateUser(email = 'demo@fitloop.ai', name = 'Demo User') {
  try {
    const listRes = await fetch(`${API_BASE}/api/v1/users`);
    if (listRes.ok) {
      const users = await listRes.json();
      if (users && users.length > 0) {
        return users[0];
      }
    }
  } catch (_) {}

  // Create default user
  const res = await fetch(`${API_BASE}/api/v1/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email }),
  });
  if (!res.ok) throw new Error('Failed to create user');
  return res.json();
}

export async function uploadWardrobeScan(imageFile, userId, scanType = 'closet') {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('user_id', userId);
  formData.append('scan_type', scanType);

  const res = await fetch(`${API_BASE}/api/v1/scans`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Wardrobe scan failed');
  }
  return res.json();
}

export async function fetchUserWardrobe(userId) {
  const res = await fetch(`${API_BASE}/api/v1/wardrobe/${userId}`);
  if (!res.ok) throw new Error('Failed to fetch wardrobe');
  return res.json();
}

export async function deleteWardrobeItem(userId, itemId) {
  const res = await fetch(`${API_BASE}/api/v1/wardrobe/${userId}/items/${itemId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete wardrobe item');
  }
  return res.json();
}

export async function updateWardrobeItem(userId, itemId, updates) {
  const res = await fetch(`${API_BASE}/api/v1/wardrobe/${userId}/items/${itemId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to update wardrobe item');
  }
  return res.json();
}

export async function runCheckBeforeBuy(imageFile, userId) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('user_id', userId);

  const res = await fetch(`${API_BASE}/api/v1/check-before-buy`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Check-before-buy evaluation failed');
  }
  return res.json();
}

export async function fetchIntegratedProfile(userId) {
  const res = await fetch(`${API_BASE}/api/v1/profile/${userId}`);
  if (!res.ok) throw new Error('Failed to fetch integrated profile');
  return res.json();
}
