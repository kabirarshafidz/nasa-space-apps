import { Button } from "@/components/ui/button";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import Link from "next/link";

// Navigation links array to be used in both desktop and mobile menus
const navigationLinks = [
  { href: "/train", label: "Train" },
  { href: "/predict", label: "Predict" },
  { href: "/3d-visualization", label: "3D Visualization" },
];

export default function NavbarNasa() {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 z-[9999] bg-background/80 backdrop-blur-md border-b border-primary/20">
      <div className="container mx-auto h-full px-4 md:px-6">
        <div className="flex h-full items-center justify-between">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <img
              src="/logo2x.png"
              alt="NASA Exoplanet Detection"
              width={90}
              height={90}
              className="h-10 w-auto"
            />
          </Link>

          {/* Desktop Navigation */}
          <NavigationMenu className="hidden md:flex">
            <NavigationMenuList className="flex items-center gap-1">
              {navigationLinks.map((link) => (
                <NavigationMenuItem key={link.href}>
                  <NavigationMenuLink
                    href={link.href}
                    className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-lg text-primary-foreground/70 hover:text-primary-foreground hover:bg-primary/10 transition-all duration-200"
                  >
                    {link.label}
                  </NavigationMenuLink>
                </NavigationMenuItem>
              ))}
            </NavigationMenuList>
          </NavigationMenu>

          {/* Mobile Menu */}
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="md:hidden hover:bg-primary/10 transition-colors"
              >
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  viewBox="0 0 24 24"
                >
                  <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
                <span className="sr-only">Toggle menu</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent
              align="end"
              className="w-48 p-2 bg-background/95 backdrop-blur-md border-primary/30"
            >
              <nav className="flex flex-col gap-1">
                {navigationLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-primary-foreground/70 hover:text-primary-foreground hover:bg-primary/10 transition-colors"
                  >
                    {link.label}
                  </Link>
                ))}
              </nav>
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </header>
  );
}
