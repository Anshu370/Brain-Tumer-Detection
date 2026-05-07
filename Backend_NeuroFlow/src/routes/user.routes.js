import { Router } from "express";
import { registerUser, loginUser, logoutUser, getUser,getDoctors} from "../controllers/user.controller.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";

const router=Router()

router.route("/register").post(registerUser)
router.route("/login").post(loginUser)
router.route("/logout").post(verifyJWT,logoutUser)
router.route("/getUser").get(getUser)
router.route("/getDoctors").get(getDoctors);

export default router