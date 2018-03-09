package utils

import config.{ServiceConfig, Settings}

object Setup {
  val settings: Settings = getSettings

  private def getSettings: Settings = {
    val serviceConf = new ServiceConfig
    Settings(Option(serviceConf.envOrElseConfig("settings.hadoopHome.value"))
      , Option(serviceConf.envOrElseConfig("settings.sparkMaster.value"))
      , Option(serviceConf.envOrElseConfig("settings.sparkAppName.value"))
      , Option(serviceConf.envOrElseConfig("settings.data1Location.value")))
  }
}
