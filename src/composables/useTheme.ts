import { ref, watch, onMounted } from "vue";

const STORAGE_KEY = "blog-theme";
type Theme = "dark" | "light";

const currentTheme = ref<Theme>("dark");

export function useTheme() {
  const toggleTheme = () => {
    currentTheme.value = currentTheme.value === "dark" ? "light" : "dark";
  };

  const setTheme = (theme: Theme) => {
    currentTheme.value = theme;
  };

  watch(currentTheme, newTheme => {
    localStorage.setItem(STORAGE_KEY, newTheme);
    applyTheme(newTheme);
  });

  const applyTheme = (theme: Theme) => {
    const html = document.documentElement;
    if (theme === "dark") {
      html.classList.add("dark");
      html.classList.remove("light");
    } else {
      html.classList.add("light");
      html.classList.remove("dark");
    }
  };

  onMounted(() => {
    const savedTheme = localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (savedTheme) {
      currentTheme.value = savedTheme;
    } else {
      // Check system preference
      const prefersDark = window.matchMedia(
        "(prefers-color-scheme: dark)"
      ).matches;
      currentTheme.value = prefersDark ? "dark" : "light";
    }
    applyTheme(currentTheme.value);
  });

  return {
    currentTheme,
    toggleTheme,
    setTheme,
    isDark: () => currentTheme.value === "dark",
  };
}
