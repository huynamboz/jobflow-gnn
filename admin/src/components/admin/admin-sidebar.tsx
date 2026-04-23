import { NavLink, useLocation } from "react-router-dom";
import { useState } from "react";
import { IconChevronDown, IconLogout, IconSettings, IconX } from "@tabler/icons-react";

import { useAuthStore } from "@/stores/auth.store";
import { adminConfig } from "@/config/admin";

const SIDEBAR_WIDTH = 272;

// Design tokens matching the batch page
const T = {
  bg:        "oklch(0.985 0.004 85)",
  surface:   "#ffffff",
  surface2:  "oklch(0.97 0.005 85)",
  line:      "oklch(0.92 0.006 85)",
  ink:       "oklch(0.18 0.02 265)",
  ink2:      "oklch(0.38 0.015 265)",
  ink3:      "oklch(0.56 0.012 265)",
  ink4:      "oklch(0.72 0.008 265)",
  accent:    "oklch(0.55 0.20 240)",
  accent600: "oklch(0.48 0.20 240)",
  accent50:  "oklch(0.97 0.03 240)",
};


export interface AdminSidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export function AdminSidebar({ isOpen = true, onClose }: AdminSidebarProps) {
  const location = useLocation();
  const logout = useAuthStore((state) => state.logout);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (title: string) =>
    setCollapsed((prev) => ({ ...prev, [title]: !prev[title] }));

  const isActive = (href: string) =>
    href === "/admin" ? location.pathname === "/admin" : location.pathname.startsWith(href);

  return (
    <>
      {isOpen && (
        <button
          aria-label="Close sidebar"
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          type="button"
          onClick={onClose}
        />
      )}

      <aside
        style={{
          width: SIDEBAR_WIDTH,
          background: T.bg,
          borderRight: `1px solid ${T.line}`,
          display: "flex",
          flexDirection: "column",
          padding: "20px 16px 24px",
          gap: 18,
          position: "fixed",
          left: 0,
          top: 0,
          height: "100vh",
          zIndex: 50,
          transition: "transform 0.2s ease-out",
        }}
        className={isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      >
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 8px 10px", fontWeight: 800, letterSpacing: "-0.02em", fontSize: 17 }}>
          <NavLink to="/admin" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none", color: T.ink }}>
            <div style={{
              width: 30, height: 30, borderRadius: 10,
              background: T.ink, color: "#fff",
              display: "grid", placeItems: "center",
              fontWeight: 800, fontSize: 14,
              position: "relative", overflow: "hidden",
            }}>
              <span style={{
                position: "absolute", inset: 0,
                background: `radial-gradient(circle at 70% 30%, ${T.accent}, transparent 60%)`,
                opacity: 0.9,
              }} />
              <span style={{ position: "relative", zIndex: 1 }}>J</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
              <span style={{ color: T.ink }}>jobflow</span>
              <span style={{ fontSize: 10.5, fontWeight: 500, color: T.ink4, marginTop: 3, letterSpacing: "0.01em" }}>HR data platform</span>
            </div>
          </NavLink>

          <button
            type="button"
            onClick={onClose}
            style={{
              marginLeft: "auto", background: "transparent", border: "none",
              cursor: "pointer", color: T.ink3, display: "flex", alignItems: "center",
            }}
            className="lg:hidden"
          >
            <IconX size={20} />
          </button>
        </div>

        {/* Nav */}
        <div style={{ flex: 1, overflow: "auto", marginRight: -6, paddingRight: 6, display: "flex", flexDirection: "column", gap: 10 }}>
          {adminConfig.navSections.map((section) => {
            const open = !collapsed[section.title];
            return (
              <div key={section.title} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <button
                  type="button"
                  onClick={() => toggleSection(section.title)}
                  style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "6px 10px 4px",
                    fontSize: 10.5, fontWeight: 700, color: T.ink4,
                    textTransform: "uppercase", letterSpacing: "0.1em",
                    background: "transparent", border: "none", cursor: "pointer",
                    textAlign: "left", width: "100%",
                  }}
                >
                  {section.title}
                  <IconChevronDown
                    size={20}
                    style={{
                      marginLeft: "auto",
                      transform: open ? "rotate(0deg)" : "rotate(-90deg)",
                      transition: "transform 0.15s",
                      color: T.ink4,
                    }}
                  />
                </button>

                {open && section.items.map((item) => {
                  const active = isActive(item.href);
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.href}
                      to={item.href}
                      onClick={() => onClose?.()}
                      style={{
                        display: "flex", alignItems: "center", gap: 10,
                        padding: "8px 10px", borderRadius: 12,
                        color: active ? T.ink : T.ink2,
                        background: active ? "rgba(0,0,0,0.07)" : "transparent",
                        fontWeight: active ? 600 : 500, fontSize: 13.5,
                        textDecoration: "none",
                        transition: "background 0.14s, color 0.14s",
                      }}
                      onMouseEnter={(e) => {
                        if (!active) {
                          e.currentTarget.style.background = "rgba(0,0,0,0.04)";
                          e.currentTarget.style.color = T.ink;
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!active) {
                          e.currentTarget.style.background = "transparent";
                          e.currentTarget.style.color = T.ink2;
                        }
                      }}
                    >
                      <Icon
                        size={16}
                        style={{ color: active ? T.ink2 : T.ink4, flexShrink: 0 }}
                      />
                      <span>{item.label}</span>
                    </NavLink>
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div style={{ borderTop: `1px solid ${T.line}`, paddingTop: 10, display: "flex", flexDirection: "column", gap: 2 }}>
          <NavLink
            to="/admin/settings"
            onClick={() => onClose?.()}
            style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 12, color: T.ink2, fontWeight: 500, fontSize: 13.5, textDecoration: "none" }}
            onMouseEnter={(e) => { e.currentTarget.style.background = T.surface2; e.currentTarget.style.color = T.ink; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = T.ink2; }}
          >
            <IconSettings size={16} style={{ flexShrink: 0 }} />
            Settings
          </NavLink>

          <button
            type="button"
            onClick={logout}
            style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "8px 10px", borderRadius: 12,
              background: "transparent", border: "none",
              color: T.ink2, fontWeight: 500, fontSize: 13.5,
              cursor: "pointer", textAlign: "left", width: "100%",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = T.surface2; e.currentTarget.style.color = T.ink; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = T.ink2; }}
          >
            <IconLogout size={16} style={{ flexShrink: 0 }} />
            Logout
          </button>
        </div>
      </aside>
    </>
  );
}
