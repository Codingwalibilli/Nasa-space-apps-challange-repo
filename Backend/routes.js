import express from "express"
import {image} from "./controller.js"
import {init} from "./controller.js"

const router = express.Router()

router.post("/myimg",image)
router.get("/",init)

export default router