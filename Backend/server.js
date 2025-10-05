import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import router from "./routes.js"
dotenv.config()

const PORT = process.env.PORT

const app = express()
app.use(cors())
app.use(express.json())
app.use("/",router)

app.listen(PORT, async() => {
    console.log(`Server Started on "${PORT}"`)
})