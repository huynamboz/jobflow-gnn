import type { LucideIcon } from "lucide-react";

import {
  LayoutDashboard,
  Tags,
  Briefcase,
  FileText,
  Settings,
} from "lucide-react";

export type AdminNavItem = {
  label: string;
  href: string;
  icon: LucideIcon;
};

export type AdminNavSection = {
  title: string;
  items: AdminNavItem[];
};

export const adminConfig = {
  name: "JobFlow",
  navSections: [
    {
      title: "GENERAL",
      items: [
        { label: "Dashboard", href: "/admin", icon: LayoutDashboard },
        { label: "Labeling", href: "/admin/labeling", icon: Tags },
        { label: "Jobs", href: "/admin/jobs", icon: Briefcase },
        { label: "CVs", href: "/admin/cvs", icon: FileText },
      ],
    },
    {
      title: "ACCOUNT",
      items: [
        { label: "Settings", href: "/admin/settings", icon: Settings },
      ],
    },
  ] as AdminNavSection[],
};
