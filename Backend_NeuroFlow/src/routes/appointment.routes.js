import { Router } from "express";
import { getUserAppointments, bookAppointment, updateAppointmentStatus } from "../controllers/appointmen.controller.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";

const router=Router()

router.route("/getUserAppointments").get(verifyJWT,getUserAppointments)
router.route("/bookAppointment").post(verifyJWT,bookAppointment)
router.patch(
    "/updateStatus/:appointmentId",
    verifyJWT,
    updateAppointmentStatus
  );

export default router