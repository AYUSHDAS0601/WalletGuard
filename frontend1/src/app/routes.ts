import { createBrowserRouter } from "react-router";
import { Home } from "./pages/Home";
import { Tracker } from "./pages/Tracker";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Home,
  },
  {
    path: "/tracker/:address",
    Component: Tracker,
  },
]);
