import { Router } from "express";
import { getReportsByUserId, createReport } from "../controllers/report.controller.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import multer from "multer";

const upload = multer({ dest: "uploads/" });

const router=Router()

router.route("/getReportsByUserId").get(verifyJWT,getReportsByUserId)
router.route("/createReport").post(verifyJWT, upload.single("files"),createReport)

export default router