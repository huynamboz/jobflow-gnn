import { Menu, Bell } from "lucide-react";
import { Button } from "@heroui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@heroui/popover";

import { useAuthStore } from "@/stores/auth.store";

export interface AdminHeaderProps {
  onMenuClick?: () => void;
}

export function AdminHeader({ onMenuClick }: AdminHeaderProps) {
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);

  return (
    <header className="flex h-16 items-center justify-between gap-4 border-b border-default-200 bg-white px-4 dark:border-default-100 dark:bg-default-50 md:px-6">
      {/* Left */}
      <div className="flex items-center gap-3">
        <Button
          isIconOnly
          aria-label="Toggle sidebar"
          className="lg:hidden"
          radius="sm"
          size="sm"
          variant="light"
          onPress={onMenuClick}
        >
          <Menu className="size-5 text-default-600" />
        </Button>
      </div>

      {/* Right */}
      <div className="flex items-center gap-1">
        <Button isIconOnly aria-label="Notifications" radius="full" size="sm" variant="light">
          <Bell className="size-[18px] text-default-500" />
        </Button>

        {user && (
          <Popover placement="bottom-end">
            <PopoverTrigger>
              <button className="ml-2 outline-none" type="button">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground cursor-pointer">
                  {(user.username || user.email || "A").charAt(0).toUpperCase()}
                </div>
              </button>
            </PopoverTrigger>
            <PopoverContent>
              <div className="px-1 py-2 w-48">
                <div className="px-2 py-1 mb-2">
                  <p className="text-sm font-semibold text-default-900 truncate">{user.username}</p>
                  <p className="text-xs text-default-500 truncate">{user.email}</p>
                  <p className="text-xs text-primary mt-0.5 capitalize">{user.role}</p>
                </div>
                <div className="border-t border-default-200 pt-2">
                  <Button
                    className="w-full justify-start text-danger"
                    color="danger"
                    size="sm"
                    variant="light"
                    onPress={logout}
                  >
                    Logout
                  </Button>
                </div>
              </div>
            </PopoverContent>
          </Popover>
        )}
      </div>
    </header>
  );
}
