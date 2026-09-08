import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import prettier from "eslint-config-prettier";
import globals from "globals";

export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "build/**",
      "node_modules/**",
      "drizzle/**",
      "patches/**",
      "vite.config.ts",
      "vitest.config.ts",
      "drizzle.config.ts",
      "client/public/__manus__/**",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["client/**/*.{ts,tsx,js,jsx}"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.es2024 },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
  {
    files: ["server/**/*.{ts,tsx,js,jsx}", "shared/**/*.{ts,tsx,js,jsx}"],
    languageOptions: {
      globals: { ...globals.node, ...globals.es2024 },
    },
  },
  {
    // shadcn/ui components legitimately co-export variant constants alongside
    // the component; the rule fires false positives on the entire family.
    files: ["client/src/components/ui/**/*.{ts,tsx}"],
    rules: {
      "react-refresh/only-export-components": "off",
    },
  },
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Tracked as warning, not error: existing `any` usage in SDK shims and
      // adapter glue is intentional; tighten file-by-file when convenient.
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },
  prettier
);
