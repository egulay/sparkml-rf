package config

case class Settings(hadoopHome: Option[String]
                    , sparkMaster: Option[String]
                    , sparkAppName: Option[String]
                    , data1Location: Option[String]) {

  def hasAllDefined: Boolean = {
    this.hadoopHome.isDefined && this.hadoopHome.get.nonEmpty &&
      this.sparkMaster.isDefined && this.sparkMaster.get.nonEmpty &&
      this.sparkAppName.isDefined && this.sparkAppName.get.nonEmpty &&
      this.data1Location.isDefined && this.data1Location.get.nonEmpty
  }

  def printMissingFields(): Unit = {
    val fields = this.getClass.getDeclaredFields
    for (field <- fields) {
      field.setAccessible(true)
      val name = field.getName
      if (field.get(this).asInstanceOf[Option[String]].get.isEmpty)
        println(s"${Console.RED}Missing entry: $name${Console.RESET}")
    }
  }

  def printAllFields(): Unit = {
    val fields = this.getClass.getDeclaredFields
    for (field <- fields) {
      field.setAccessible(true)
      println(s"${Console.RED}${Console.BOLD}${field.getName}:${Console.RESET} ${field.get(this).asInstanceOf[Option[String]].get}")
    }
  }
}
