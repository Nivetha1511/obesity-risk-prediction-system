const CACHE_NAME = "obesicare-v1";
const urlsToCache = [
  "/",
  "/index.html",
  "/login.html",
  "/form.html",
  "/result.html",
  "/style.css",
  "/script.js",
  "/manifest.json",
];

// Install event - cache assets
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("Opened cache");
      return cache.addAll(urlsToCache).catch((err) => {
        console.warn("Cache addAll error (some assets may not be available offline):", err);
      });
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log("Deleting old cache:", cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Skip cross-origin requests
  if (!request.url.startsWith(self.location.origin)) {
    return;
  }

  // For API calls, always try network first
  if (request.url.includes("/predict") || request.url.includes("onrender.com")) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Clone and cache successful API responses
          if (response.ok) {
            const cache = caches.open(CACHE_NAME);
            cache.then((c) => c.put(request, response.clone()));
          }
          return response;
        })
        .catch(() => {
          // Fallback to cache if network fails
          return caches.match(request);
        })
    );
    return;
  }

  // For static assets, cache-first strategy
  event.respondWith(
    caches.match(request).then((response) => {
      return (
        response ||
        fetch(request)
          .then((networkResponse) => {
            // Cache new assets
            if (networkResponse && networkResponse.status === 200) {
              const cache = caches.open(CACHE_NAME);
              cache.then((c) => c.put(request, networkResponse.clone()));
            }
            return networkResponse;
          })
          .catch(() => {
            // Fallback offline page if needed
            return new Response("Offline - please check your internet connection.", {
              status: 503,
              statusText: "Service Unavailable",
              headers: new Headers({
                "Content-Type": "text/plain",
              }),
            });
          })
      );
    })
  );
});
